import streamlit as st
import os
import base64
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.prediction import PredictionPipeline

# Function to convert image to base64 string
def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# App title
st.title("🫘Kidney Disease Classification")

# Sidebar options
app_mode = st.sidebar.selectbox("Choose the mode", ["Predict", "Train"])

if app_mode == "Predict":
    st.header("Upload an Image for Prediction")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = "inputImage.jpg"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Encode and decode image
        encoded_image = encode_image(file_path)
        decodeImage(encoded_image, file_path)

        # Run prediction pipeline
        prediction_pipeline = PredictionPipeline(file_path)
        result = prediction_pipeline.predict()

        st.image(file_path, caption="Uploaded Image", use_column_width=True)
        st.subheader("Prediction:")
        st.write(result[0]["image"])

        st.subheader("Prediction Probabilities:")
        # Assuming binary classification [Normal, Tumor], show nicely:
        probs = result[0]["probabilities"]
        st.write(f"Normal: {probs[0]:.4f}")
        st.write(f"Tumor: {probs[1]:.4f}")
