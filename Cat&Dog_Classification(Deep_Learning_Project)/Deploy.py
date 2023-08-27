import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('./my_model.h5')

def predict_image(image):
    # Preprocess the image

    #image = cv2.imread(image)
    image = image/255.
    print(image.shape)
    image = cv2.resize(image, (256, 256))
    image = image.reshape((1, 256, 256, 3))
    print(image.shape)

    prediction = model.predict(image)

    if prediction[0][0] < 0.5:
        return "Cat"
    else:
        return "Dog"

def main():
    st.title("Cat or Dog Classifier")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Predict the image
        prediction = predict_image(image)
        st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
