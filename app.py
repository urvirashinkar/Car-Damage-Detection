import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Define IMG_SIZE and preprocess_input as they were in the Colab notebook
IMG_SIZE = (224, 224)
# Ensure preprocess_input matches the one used during training (from ResNet50)
from tensorflow.keras.applications.resnet50 import preprocess_input

st.title('Car Damage Detection App')

@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_model():
    model_path = 'car_damage_model_final.keras' # Assuming you've placed the .keras file in your app directory
    model = keras.models.load_model(model_path)
    return model

model = load_model()

CLASS_NAMES = ['00-damage', '01-whole'] # Or however you map your class names
SHORT_NAMES = [c.split('-', 1)[-1].replace('-', ' ').title() for c in CLASS_NAMES]

st.write("Upload an image to detect car damage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_resized = image.resize(IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    processed_img = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = SHORT_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    st.write(f"Prediction: **{predicted_class_name}**")
    st.write(f"Confidence: **{confidence:.2f}%**")