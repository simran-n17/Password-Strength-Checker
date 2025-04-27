import streamlit as st
import joblib
from utils.preprocess import extract_features

model = joblib.load('models/password_model.pkl')
strength_map = {0: "Weak", 1: "Medium", 2: "Strong"}

st.title("Password Strength Checker")
password = st.text_input("Enter a password:", type="password")

if password:
    features = extract_features(password)
    pred = model.predict(features)[0]
    st.progress((pred + 1) / 3, text=strength_map[pred])