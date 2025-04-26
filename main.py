# app.py

import streamlit as st
import os
import pandas as pd
import joblib
from datetime import datetime
from processing import process_uploads_folder

# === Streamlit App ===

# Page configuration
st.set_page_config(page_title="Real-Time Gesture Classifier 🚀", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🖐️ Real-Time Gesture Classifier App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Upload section
st.subheader("📂 Upload your EMG Data (.csv or .txt)")

uploaded_file = st.file_uploader("Choose your file:", type=["csv", "txt"])

# Load model
model_path = "saved_rf_model.joblib"  # Update if your model has a different name
scaler, model = joblib.load(model_path)

# Ensure 'uploads' directory exists
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

if uploaded_file is not None:
    # Clear previous uploads
    for f in os.listdir(upload_dir):
        os.remove(os.path.join(upload_dir, f))

    # Save new file
    file_ext = uploaded_file.name.split('.')[-1]
    filename = f"uploaded.{file_ext}"
    filepath = os.path.join(upload_dir, filename)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File uploaded successfully: `{filename}`")

    # Process and Predict
    st.subheader("🔍 Processing and Predicting...")

    with st.spinner('🔄 Please wait... Processing your file and predicting gestures...'):
        data_to_predict = process_uploads_folder(upload_dir)

        if data_to_predict is not None:
            try:
                # Scale the features before prediction
                data_to_predict_scaled = scaler.transform(data_to_predict)
                predictions = model.predict(data_to_predict_scaled)

                result_df = pd.DataFrame({"Predicted Gesture": predictions})

                st.success("🎯 Prediction Completed!")

                st.dataframe(result_df, use_container_width=True)

                # Download button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="predicted_gestures.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.error("❌ Failed to process data. Make sure your file is correctly formatted.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
