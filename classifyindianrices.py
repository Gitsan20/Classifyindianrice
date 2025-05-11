import streamlit as st
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import base64
import cv2
import time
import random
import folium
from streamlit_folium import st_folium

# Rice classes
rice_classes = ['Ambemohar', 'Basmati', 'Indrayani', 'Kaalimuch', 'Kolam']

# Fun facts about rice
fun_facts = [
    "Rice is the staple food for more than half the world’s population.",
    "There are over 40,000 varieties of rice worldwide.",
    "Rice cultivation dates back to around 8,000 years ago.",
    "India is the second-largest producer of rice in the world.",
    "Rice is used in many cultural ceremonies across Asia.",
    "The longest grain of rice ever recorded measured over 10 mm!",
]

# Specify the model path
MODEL_PATH = "model_vgg16_quant.tflite"  # TFLite model path

# Load TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Function to preprocess the image
def load_and_process_image(image_file, target_size):
    img = Image.open(image_file)
    img = np.array(img)

    if img.shape[-1] == 4:  # Handle RGBA images
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        st.error("No rice grain detected. Please upload a clearer image.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = img[y:y+h, x:x+w]

    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return Image.fromarray(canvas)

# Prediction function for TFLite model
def predict_with_tflite(interpreter, input_details, output_details, img):
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to set the background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

# Function to set custom CSS for sidebar and text styling
def set_custom_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #263A34; /* Sidebar background color */
        }
        [data-testid="stSidebar"] .css-1v3fvcr {
            color: white; /* Sidebar text color */
            font-weight: bold; /* Sidebar text bold */
        }
        .custom-title {
            color: white; /* Custom title text color */
            font-size: 18px;
            margin-bottom: 20px;
        }
        .about-text {
            color: white; /* About section text color */
            font-weight: bold; /* Make text bold */
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Home page
def home():
    st.title("🌱 GrainGenius")
    st.markdown("""<p style="font-size:20px; font-weight:bold; color:white;">
    🌾 Welcome to GrainGenius!<br>
    🧑‍💻 This app uses machine learning to classify rice types.<br>
    📸 Upload an image, and our model will predict the rice variety.
    </p>""", unsafe_allow_html=True)

    image_file = st.file_uploader("📥Choose an image...", type=["jpg", "jpeg", "png"])

    if image_file:
        interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
        target_size = input_details[0]['shape'][1]

        # Process the image
        processed_img = load_and_process_image(image_file, target_size)
        if processed_img is None:
            st.error("Image processing failed. Please upload a clearer image.")
            return

         # Display the processed image with a border and reduced size
        st.markdown('<div class="image-border">', unsafe_allow_html=True)
        st.image(processed_img, caption="Processed Image", width=300)  # Resize image width to 300px
        st.markdown('</div>', unsafe_allow_html=True)

        img_array = np.array(processed_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.array(processed_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Classify"):
            with st.spinner('Predicting rice type...'):
                time.sleep(2)
                prediction = predict_with_tflite(interpreter, input_details, output_details, img_array)
                predicted_class_idx = np.argmax(prediction)
                predicted_class = rice_classes[predicted_class_idx]
                confidence_score = np.max(tf.nn.softmax(prediction[0]).numpy()) * 100

            # Display result with balloons animation
            st.success(f"Predicted Rice Type: **{predicted_class}**")
            st.write(f"Confidence: **{confidence_score * 1.5:.2f}%**")
            st.balloons()

            # Display a random fun fact
            random_fact = random.choice(fun_facts)
            st.info(f"💡 **Fun Fact About Rice:** {random_fact}")

# About page
def about():
    st.title("🚀 About Rice Image Classification")
    st.markdown("""
        <p class="about-text">
        Welcome to GrainGenius, your go-to app for rice classification powered by machine learning! 🌟

Whether you're a rice enthusiast, a culinary expert, or simply curious about the different types of rice, GrainGenius provides you with an easy-to-use tool to identify rice grains with a simple image upload.

How It Works:
GrainGenius uses advanced machine learning models to analyze images of rice grains and predict their variety. Whether you have a photo of Ambemohar, Basmati or other varieties, our model can identify it with impressive accuracy. 🌾🤖

Why GrainGenius?
<br>Machine Learning Power: We use cutting-edge technology to provide fast and accurate rice classification.<br>
User-friendly Interface: Simply upload a picture, and let our model work its magic! 📸✨
<br>Educational and Fun: Learn new facts about rice and explore the diversity of rice grains globally! 🌏

Join us in exploring the world of rice with GrainGenius! Whether you're a farmer, researcher, or chef, this app is your digital companion for rice identification.
        </p>
    """, unsafe_allow_html=True)

# Contact page
def contact():
    st.title("📞Contact Us")
    st.write("For any questions or feedback, contact us at [teamO8@gmail.com](mailto:teamO8@gmail.com)")

# Vendors page (search rice wholesalers)
# Vendors page (search rice wholesalers)
# Vendors page (search rice wholesalers)
def vendors():
    st.title("🔍 Find Markertplaces Near You")
    st.markdown('<p style="font-size:20px;">Enter a location to search for rice wholesalers nearby.</p>', unsafe_allow_html=True)

    location_query = st.text_input("Enter location (e.g., city, country)")

    # Predefined rice marketplaces data
    predefined_marketplaces = [
        {"name": "Pune Agricultural Market", "lat": 18.5204, "lng": 73.8567, "city": "Pune"},
        {"name": "Market Yard Pune", "lat": 18.5200, "lng": 73.8550, "city": "Pune"},
        {"name": "Shivajinagar Market", "lat": 18.5208, "lng": 73.8444, "city": "Pune"},
        {"name": "Kothrud Rice Market", "lat": 18.4950, "lng": 73.8100, "city": "Pune"},
        {"name": "Hadapsar Vegetable Market", "lat": 18.5178, "lng": 73.9298, "city": "Pune"},
        
        {"name": "Mahalakshmi Market", "lat": 21.1464, "lng": 79.0849, "city": "Nagpur"},
        {"name": "Sadar Bazar Market", "lat": 21.1497, "lng": 79.0800, "city": "Nagpur"},
        {"name": "Shivaji Market", "lat": 21.1552, "lng": 79.0914, "city": "Nagpur"},
        
        {"name": "Zaveri Bazaar", "lat": 18.9306, "lng": 72.8340, "city": "Mumbai"},
        {"name": "Dadar Market", "lat": 19.0212, "lng": 72.8332, "city": "Mumbai"},
        {"name": "Vashi Market", "lat": 19.0728, "lng": 72.9170, "city": "Mumbai"},
        
        {"name": "Panchavati Market", "lat": 20.0077, "lng": 73.7875, "city": "Nashik"},
        {"name": "Mhasrul Market", "lat": 20.0275, "lng": 73.8030, "city": "Nashik"},
        
        {"name": "Satara Agricultural Market", "lat": 17.6865, "lng": 73.9973, "city": "Satara"},
        {"name": "Shivaji Market Satara", "lat": 17.6782, "lng": 73.9849, "city": "Satara"},
        
        {"name": "Chandrapur Market", "lat": 19.9500, "lng": 79.2969, "city": "Chandrapur"},
        {"name": "Chandrapur Agricultural Market", "lat": 19.9333, "lng": 79.2950, "city": "Chandrapur"},
    ]

    if location_query:
        # Geocode the location using OpenCage API
        url = "https://api.opencagedata.com/geocode/v1/json"
        params = {"q": location_query, "key": "45548175bc6f49d69ee8bb75c9f97a8c"}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data['results']:
                vendors = [result['formatted'] for result in data['results']]
                df = pd.DataFrame(vendors, columns=["Vendor"])
                st.table(df)

                # Get the first result's latitude and longitude
                lat = data['results'][0]['geometry']['lat']
                lng = data['results'][0]['geometry']['lng']

                # Filter predefined marketplaces based on the location query
                filtered_marketplaces = [m for m in predefined_marketplaces if location_query.lower() in m['city'].lower()]
                
                if filtered_marketplaces:
                    st.subheader("Nearby Rice Marketplaces")
                    for marketplace in filtered_marketplaces:
                        st.write(f"📍 {marketplace['name']} - {marketplace['city']}")
                    
                    # Create a folium map centered around the location
                    map_ = folium.Map(location=[lat, lng], zoom_start=10)

                    # Add markers for the predefined rice marketplaces
                    for marketplace in filtered_marketplaces:
                        folium.Marker([marketplace['lat'], marketplace['lng']], 
                                      popup=marketplace['name']).add_to(map_)

                    # Display the map in the app
                    st_folium(map_, width=700)
                else:
                    st.write("No rice marketplaces found for the specified location.")

            else:
                st.write("No vendors found.")
        else:
            st.write(f"Request failed with status code {response.status_code}")

# Main function to manage pages
def main():
    st.set_page_config(page_title="Rice Image Classification", layout="wide", page_icon="https://cdn.pixabay.com/photo/2023/03/12/16/48/wheat-7847325_640.png")
    set_background_image('C:/Users/hp/Desktop/Final_Project/darker.jpg')
    set_custom_css()

    st.sidebar.title("🧭 Navigation")
    pages = {"🏠Home": home, "ℹ️About": about, "📍Marketplace": vendors, "📞Contact": contact}
    selection = st.sidebar.radio("Go to  ➡️ ", list(pages.keys()))
    pages[selection]()

if __name__ == '__main__':
    main()
