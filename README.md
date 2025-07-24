# Plant Disease Detection

# 🌿 Plant Disease Detection System for Sustainable Agriculture

This project is a Deep Learning-based web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) trained in TensorFlow. The application is deployed using Streamlit on Hugging Face Spaces.

---

## 📌 Features

- 🌱 Classifies plant leaf images into multiple disease categories
- 🧠 TensorFlow-based CNN model trained on plant disease dataset
- 📦 Saved model loaded into a Streamlit app
- 🐳 Dockerized and deployed on Hugging Face Spaces
- 🖼️ Accepts uploaded images for real-time prediction

---

## 📂 Dataset

We used the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) available on Kaggle.

📁 Dataset Summary:
- Over 50,000 labeled images of diseased and healthy plant leaves
- 38 plant classes including tomato, potato, corn, apple, etc.

📥 Download:
```bash
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
```

---

## 🧠 Model Training

The model is trained in this Jupyter Notebook:  
👉 [`Plant_Disease_Detection_System_for_Sustainable_Agriculture.ipynb`](./Plant_Disease_Detection_System_for_Sustainable_Agriculture.ipynb)

### 🔧 Key Details:
- Framework: TensorFlow 2.x
- Model: CNN with Conv2D, MaxPooling2D, Dense layers
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Accuracy: ~99% on training set and ~97% on validation set
- Saved as: `plant_disease_detection_model.keras`

---

## 🖥️ Live Web App

🔗 **Live on Hugging Face Spaces**:  
📌 `https://huggingface.co/spaces/nagabhushan-gullipalli/plant-disease-detection`

---

## 🚀 Run the App Locally

```bash
git clone https://github.com/gullipalli-nagabhushan/plant-disease-detection.git
cd plant-disease-detection

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py
```

---

## 🐳 Docker Deployment (used for Hugging Face)

```dockerfile
FROM python:3.10-slim
WORKDIR /code
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install tensorflow==2.15.0
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

---

## 📁 Project Structure

```
├── app.py                            # Streamlit web app
├── plant_disease_model.h5           # Trained model
├── requirements.txt
├── Dockerfile
├── README.md
└── Plant_Disease_Detection_System_for_Sustainable_Agriculture.ipynb
```

---

## ✨ Future Improvements

- Add Grad-CAM visualizations for explainability
- Improve mobile responsiveness
- Use ONNX for cross-platform deployment
- Add disease-specific treatment suggestions

---

## 📧 Contact

Created by [Gullipalli Nagabhushan](mailto:gullipallinagabhushan@gmail.com)

---

## 📜 License

MIT License – feel free to use, share, and improve.
