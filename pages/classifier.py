import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'my_model.hdf5' 
model = load_model(model_path)

st.write("# Flower Classification")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Ensure these match your model's classes

def import_and_predict(image_data, model):
    size = (180, 180)    
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)  # Updated for new Pillow version
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure color format matches your model's training
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', width=250)  # Adjust the width as needed

    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
