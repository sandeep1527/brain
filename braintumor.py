import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the pre-trained brain tumor classification model
def load_model():
    model = tf.keras.models.load_model('brain.h5')
    return model
 
model = load_model()

st.title('Brain Tumor Prediction')

# Function to make a prediction
def predict_class(image, model):
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image to the model's input shape
    resized_image = cv2.resize(np.array(image), (150, 150))
    reshaped_image = np.reshape(resized_image, (1, 150, 150, 3))
    
    # Make a prediction
    predictions = model.predict(reshaped_image)
    pred = predictions[0][0]
    
    # Interpret the result
    if pred > 0.5:
        result = "HEALTHY"
    else:
        result = "BRAIN TUMOR"
    return result

# File uploader for image input
file = st.file_uploader("Upload an MRI scan image", type=["jpg", "png", "jpeg", "tif"])

if file is None:
    st.text('Waiting for image upload...')
else:
    # Display the input image
    test_image = Image.open(file)
    st.image(test_image, caption="Uploaded MRI Image", width=400)
    
    # Convert image to a numpy array and make prediction
    result = predict_class(test_image, model)
    
    # Display the prediction result
    st.success(f'The model predicts: {result}')
