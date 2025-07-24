import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home.png"
    st.image(image_path, use_container_width=True)  # Updated parameter here
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to make plant disease identification faster, easier, and more accurate. Upload an image of your plant, and let our system analyze it to detect potential diseases. Together, we can protect our crops and promote healthy farming practices!

    ### How It Works
    1. **Upload Image:** Visit the **Disease Recognition** page and upload an image of your plant that shows any signs of illness.
    2. **Analysis:** Our advanced machine learning model will evaluate the image to identify any diseases.
    3. **Results:** View the predictions along with recommendations on how to treat or prevent the disease.

    ### Why Choose Us?
    - **Highly Accurate:** Our system leverages cutting-edge AI to ensure precise disease detection.
    - **Easy-to-Use:** Intuitive interface designed for everyone, from farmers to researchers.
    - **Quick Results:** Get instant predictions, helping you take action right away.

    ### Get Started
    Head to the **Disease Recognition** page from the sidebar to upload an image and see how our system works in real time!

    ### About Us
    Curious about the project, the team behind it, and our goals? Head over to the **About** page to learn more.
    """)


#About Project
elif(app_mode=="About"):
    st.header("About the Project")
    st.markdown("""
                #### Dataset Overview
                This dataset has been recreated through offline augmentation from the original dataset. You can explore the original data on GitHub.
                It contains approximately 87,000 RGB images of healthy and diseased crop leaves, divided into 38 categories. The dataset is split into training and validation sets, with 80% allocated for training and 20% for validation. Additionally, a new directory with 33 test images was created for disease prediction purposes.
                
                #### Dataset Breakdown
                1. Training Set (70,295 images)
                2. Test Set (33 images)
                3. Validation Set (17,572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Upload Your Plant Image:")
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_container_width=True)  # Updated parameter here
    
    #Prediction button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction:")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        result = class_name[result_index]
        result = result.replace('_', ' ')
        st.success("The model predicts that your plant is: **{}**".format(result))
