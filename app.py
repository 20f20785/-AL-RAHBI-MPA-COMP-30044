import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define the correct order of feature columns (must match training)
feature_names = [
    'brand', 'processor', 'Ram', 'Ram_type', 'ROM', 'ROM_type', 'GPU',
    'display_size', 'resolution_width', 'resolution_height', 'OS', 'warranty', 'spec_rating'
]

st.title("💻 Laptop Price Estimator")

# Input fields
brand = st.selectbox("Brand", label_encoders['brand'].classes_)
processor = st.selectbox("Processor", label_encoders['processor'].classes_)
ram = st.slider("RAM (GB)", 4, 64, 8)
ram_type = st.selectbox("RAM Type", label_encoders['Ram_type'].classes_)
rom = st.slider("ROM (GB)", 128, 2048, 512)
rom_type = st.selectbox("ROM Type", label_encoders['ROM_type'].classes_)
gpu = st.selectbox("GPU", label_encoders['GPU'].classes_)
display_size = st.slider("Display Size (inches)", 11.0, 18.0, 15.6)
res_w = st.number_input("Resolution Width", 1280, 3840, 1920)
res_h = st.number_input("Resolution Height", 720, 2160, 1080)
os = st.selectbox("Operating System", label_encoders['OS'].classes_)
warranty = st.selectbox("Warranty (Years)", [0, 1, 2, 3])
spec_rating = st.slider("Spec Rating", 50.0, 90.0, 70.0)

# Create dictionary from inputs
input_data = {
    'brand': label_encoders['brand'].transform([brand])[0],
    'processor': label_encoders['processor'].transform([processor])[0],
    'Ram': ram,
    'Ram_type': label_encoders['Ram_type'].transform([ram_type])[0],
    'ROM': rom,
    'ROM_type': label_encoders['ROM_type'].transform([rom_type])[0],
    'GPU': label_encoders['GPU'].transform([gpu])[0],
    'display_size': display_size,
    'resolution_width': res_w,
    'resolution_height': res_h,
    'OS': label_encoders['OS'].transform([os])[0],
    'warranty': warranty,
    'spec_rating': spec_rating
}

# Convert to DataFrame and reorder columns
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_names]

# 🔧 FIXED: Use .values to avoid feature name mismatch
scaled_input = scaler.transform(input_df.values)

# Prediction
if st.button("Estimate Price"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated Laptop Price: ₹{prediction * 100000:.2f}")
