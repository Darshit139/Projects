import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('./my_model.h5')

def predict_image(image):
    # Preprocess the image
    image = cv2.imread(image)
    image = image/255.
    print(image.shape)
    image = cv2.resize(image, (256, 256))
    image = image.reshape((1, 256, 256, 3))
    print(image.shape)

    prediction = model.predict(image)

    if prediction[0][0] < 0.5:
        print(prediction)
        print("Cat")
    else:
        print(prediction)
        print("Dog")

predict_image('./dog.3.jpg')
