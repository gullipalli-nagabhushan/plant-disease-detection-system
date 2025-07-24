import streamlit as st
import tensorflow as tf
import numpy as np

# --- Model Prediction ---
def model_prediction(test_image):
    model = tf.keras.models.load_model("plant_disease_detection_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# --- Sidebar Navigation ---
st.sidebar.title("🌱 Plant Disease Dashboard")
app_mode = st.sidebar.selectbox("Navigate", ["Home", "About", "Disease Recognition"])

# --- Home Page ---
if app_mode == "Home":
    st.title("🌿 Plant Disease Detection System")
    st.image("home.png", use_container_width=True)
    st.markdown("""
Welcome to the **Plant Disease Recognition System**!  
Protect crops. Diagnose early. Improve yield. 🌾

### 🔍 How It Works
1. **Upload** a clear image of the plant leaf.
2. The AI model will **analyze** the image.
3. You get **instant prediction** of the disease type!

### 📊 Features
- Fast & accurate predictions
- Simple UI with Streamlit
- Deployed using Docker on Hugging Face

### 🚀 Live Demo & Resources
- 🌐 [View Live on Hugging Face](https://huggingface.co/spaces/gullipalli-nagabhushan/plant-disease-detection)
- 💾 [GitHub Repository](https://github.com/gullipalli-nagabhushan/plant-disease-detection-system)
- 📚 [Dataset (PlantVillage on Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
""")

# --- About Page ---
elif app_mode == "About":
    st.title("📘 About the Project")
    st.markdown("""
This AI-powered system uses a deep learning model trained on the **PlantVillage dataset** containing ~87,000+ leaf images across 38 disease and healthy classes.

### 🧪 Dataset Summary:
- 🔹 70,295 training images  
- 🔹 17,572 validation images  
- 🔹 33 test images

This system was built to empower **smart farming**, **reduce crop loss**, and provide **real-time insights** to farmers and researchers alike.
""")

# --- Disease Recognition Page ---
elif app_mode == "Disease Recognition":
    st.title("🧠 Disease Recognition")
    test_image = st.file_uploader("📤 Upload a Plant Leaf Image")

    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            st.spinner("Analyzing the image...")
            result_index = model_prediction(test_image)

            # Disease class mapping
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]

            prediction = class_names[result_index].replace('_', ' ')
            st.success(f"🧬 Predicted Disease: **{prediction}**")

# --- Footer ---
st.markdown("---")
st.markdown("🔗 [GitHub](https://github.com/gullipalli-nagabhushan/plant-disease-detection-system) | 🤗 [Hugging Face Space](https://huggingface.co/spaces/gullipalli-nagabhushan/plant-disease-detection) | 📊 [Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)", unsafe_allow_html=True)
