# Plant Disease Detection

## Project Overview
This project aims to develop an AI-based solution for detecting plant diseases using computer vision and machine learning. It leverages deep learning techniques to identify common plant diseases from images of plant leaves. The goal is to assist farmers and agricultural professionals in early disease detection to improve crop yield and reduce pesticide usage.

## Features
- **Disease Detection**: Detects various plant diseases based on leaf images.
- **Real-time Analysis**: Provides quick diagnosis using pre-trained models for plant diseases.
- **Mobile Compatibility**: The system can be used in mobile applications for easy access in field conditions.
- **Predictive Analytics**: Suggests preventive measures based on detected diseases.
  
## Technologies Used
- **Python**: Programming language for implementing machine learning and data processing.
- **TensorFlow/Keras**: For building and training the deep learning model.
- **OpenCV**: For image processing and manipulation.
- **Streamlit**:  For building web services.

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/gullipalli-nagabhushan/plant-disease-detection-system.git
2. Navigate to the project folder:
    ```bash
    cd plant-disease-detection
3. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    ```

    **a. For macOS/Linux:**
    ```bash
    source .venv/bin/activate
    ```

    **b. For Windows:**
    ```bash
    .venv\Scripts\activate
    ```
 

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
5. Run the application:
    ```bash
    streamlit run main.py  #  to launch the server
    
## Usage
- **Input**: Upload images of plant leaves through the web interface.
- **Output**: The model will predict the disease based on the leaf's appearance.

The model was trained on a dataset containing labeled images of various plant diseases. You can explore the dataset used in the project here:
<a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset" target="_blank">Link to Dataset.</a>

## Future Work
- **Expand Dataset**: Incorporate more images of plant diseases to improve accuracy and generalization across different plant species.
- **Real-time Suggestions**: Upon detection, the system will display suggestions for treatment or management.
- **Real-Time Monitoring**: Implement live camera feeds and IoT sensor data integration for real-time disease detection in field conditions.
- **Mobile App**: Develop a fully functional mobile application to provide instant diagnosis in the field using phone cameras.
- **Advanced AI Models**: Experiment with more advanced deep learning architectures like GANs or Reinforcement Learning for enhanced disease prediction.

## Contributing
We welcome contributions from the community! If you want to contribute, please fork the repository and submit a pull request. Make sure to follow the contribution guidelines.

## License
This project is licensed under the MIT License - see the LICENSE file for more information.

## Acknowledgments
- TensorFlow for providing powerful machine learning tools.
- OpenCV for offering image processing capabilities.
- Kaggle Plant Disease Dataset for supplying the dataset used to train the model.
- Streamlit for the web framework.




### Highlights of the `README.md` file:

1. **Project Description**: An overview of the plant disease detection system and its goals.
2. **Features**: A list of the key features the system provides.
3. **Technologies Used**: Describes the technologies, libraries, and frameworks powering the system.
4. **Installation Guide**: Step-by-step instructions for setting up the project locally.
5. **Usage**: A brief explanation of how users interact with the system.
6. **Dataset**: Information about the dataset used for training the model.
7. **Future Work**: Ideas for how the project can be expanded and improved.
8. **Contributing**: Encourages external contributions to the project.
9. **License**: Information about the license governing the project.
10. **Acknowledgments**: Credits to the libraries and resources used.

This version is formatted for clarity, and you can adapt it to your specific project details as needed.
