import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

model = keras.models.load_model("digit_model.h5")

st.title("Number Drawing Classifier")

uploaded_file = st.file_uploader("**Upload a Digit Drawing Image**", type=["png"])

button_pressed = st.button("Predict")

if uploaded_file is None and button_pressed:
    st.error("No Image Uploaded")

if uploaded_file is not None and button_pressed:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28,28))
    
    img = np.array(img) / 255.0
    display_img = img.copy()

    if img.mean() > 0.7:
        img = 1 - img

    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    
    st.success("Prediction made!")
    st.image(display_img.reshape(28,28), caption="Your Image")
    st.write("Your number is:", prediction.argmax())