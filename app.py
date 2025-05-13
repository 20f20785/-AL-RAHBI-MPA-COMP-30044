
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and preprocessing objects
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoder_dict.pkl", "rb"))

# Title
st.title("💻 Laptop Price Estimator")

# User Inputs
brand = st.selectbox("Brand", encoders['brand'].classes_)
processor = st.selectbox("Processor", encoders['processor'].classes_)
ram = st.slider("RAM (GB)", 4, 64, 8)
ram_type = st.selectbox("RAM Type", encoders['Ram_type'].classes_)
rom = st.slider("ROM (GB)", 128, 2048, 512)
rom_type = st.selectbox("ROM Type", encoders['ROM_type'].classes_)
gpu = st.selectbox("GPU", encoders['GPU'].classes_)
display_size = st.slider("Display Size (inches)", 11.0, 18.0, 15.6)
res_w = st.number_input("Resolution Width", 1280, 3840, 1920)
res_h = st.number_input("Resolution Height", 720, 2160, 1080)
os = st.selectbox("Operating System", encoders['OS'].classes_)
warranty = st.selectbox("Warranty (Years)", [0, 1, 2, 3])
spec_rating = st.slider("Spec Rating", 50.0, 90.0, 70.0)

# Convert input to DataFrame
input_df = pd.DataFrame({
    'brand': [encoders['brand'].transform([brand])[0]],
    'processor': [encoders['processor'].transform([processor])[0]],
    'Ram': [ram],
    'Ram_type': [encoders['Ram_type'].transform([ram_type])[0]],
    'ROM': [rom],
    'ROM_type': [encoders['ROM_type'].transform([rom_type])[0]],
    'GPU': [encoders['GPU'].transform([gpu])[0]],
    'display_size': [display_size],
    'resolution_width': [res_w],
    'resolution_height': [res_h],
    'OS': [encoders['OS'].transform([os])[0]],
    'warranty': [warranty],
    'spec_rating': [spec_rating]
})

# Scale inputs
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Estimate Price"):
    price = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated Laptop Price: ₹{price * 100000:.2f}")
