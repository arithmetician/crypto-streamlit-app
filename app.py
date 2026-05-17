"""
Cryptocurrency Market Prediction Streamlit App
"""

import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# Google Drive file ID
file_id = "1giq04u4fiViX8U1bDx3VVYoWosGuHU-c"

# Download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download model if it doesn't exist
if not os.path.exists("crypto_model.pkl"):
    gdown.download(url, "crypto_model.pkl", quiet=False)

# Load model
model = joblib.load("crypto_model.pkl")

# Title
st.title("Cryptocurrency Market Prediction App")

# Description
st.write("Predict cryptocurrency closing prices using machine learning.")

# Sidebar
st.sidebar.header("Input Features")

# Inputs
open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0.0)
ma7 = st.number_input("7-Day Moving Average", min_value=0.0)
ma30 = st.number_input("30-Day Moving Average", min_value=0.0)
daily_return = st.number_input("Daily Return")
volatility = st.number_input("Volatility", min_value=0.0)

# Predict Button
if st.button("Predict"):

    try:

        input_data = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Volume': [volume],
            'MA_7': [ma7],
            'MA_30': [ma30],
            'Daily_Return': [daily_return],
            'Volatility': [volatility]
        })

        prediction = model.predict(input_data)

        predicted_price = float(prediction[0])

        st.success(f"Predicted Close Price: ${predicted_price:,.2f}")

    except (ValueError, TypeError) as e:

        st.error(f"Error: {e}")

