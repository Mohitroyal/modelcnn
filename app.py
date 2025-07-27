import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
from tensorflow.keras.models import load_model

# Load model
model = load_model("cnn.h5")



# Load the model
model = load_model("cnn.h5")

# Class labels
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Streamlit UI
st.title("CIFAR-10 Image Classifier")
uploaded_file = st.file_uploader("Upload a 32x32 image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.write("Prediction:", classes[class_index])
    st.write("Confidence:", f"{confidence:.2%}")
