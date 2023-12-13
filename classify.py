#this is the initial approach we used to implement vector search but due to google cloud credits being all used up, we had to move to a different approach.


import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from tensorflow.keras.models import load_model



def make_prediction(image_data, model):
    size = (180, 180)    
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)  # Updated for new Pillow version
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure color format matches your model's training
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    score = tf.nn.softmax(predictions[0])
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Ensure these match your model's classes
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    answer = f'Prediction: {predicted_class} \n Confidence: {confidence:.2f}%'
    return answer
