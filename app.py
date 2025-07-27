# app.py
import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2

# Load ONNX model
session = ort.InferenceSession("cnn.h5")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

st.title("CIFAR-10 Image Classifier")
uploaded_file = st.file_uploader("Upload an image (32x32)", type=["jpg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Remove alpha channel if exists

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype(np.float32)  # Ensure float32

    # Predict using ONNX Runtime
    prediction = session.run([output_name], {input_name: img_array})[0]
    class_index = np.argmax(prediction)
    st.write("Prediction:", classes[class_index])
