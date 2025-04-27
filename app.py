import streamlit as st
import os
from model.load_model import get_trained_model
from utils.file_manager import manage_uploads_folder, save_uploaded_file
from utils.feature_extraction import preprocess_uploaded_data
import pandas as pd

# === Streamlit Page Setup ===
st.set_page_config(page_title="Gesture Prediction System ✋", page_icon="🧠", layout="centered")

st.title("🤖 Real-Time EMG Gesture Recognition")
st.write("Upload your EMG data file to detect hand gestures instantly!")

# === Uploading Section ===
st.header("📤 Upload EMG Data File")
data_file = st.file_uploader("Drag and Drop your CSV or TXT file below:", type=["csv", "txt"])

# Load pre-trained model and scaler
scaler, classifier = get_trained_model()

# Make sure the upload directory is clean
UPLOAD_DIR = "uploadsFolder"
manage_uploads_folder(UPLOAD_DIR)

# When a file is uploaded
if data_file is not None:
    # Save file
    file_path = save_uploaded_file(data_file, UPLOAD_DIR)
    st.success(f"✅ File `{data_file.name}` uploaded successfully!")

    # Data Processing and Prediction
    st.header("🔎 Data Processing and Prediction")
    with st.spinner('Processing your data, please wait...'):
        processed_data = preprocess_uploaded_data(file_path)

        if processed_data is not None:
            try:
                # Scaling features before prediction
                scaled_features = scaler.transform(processed_data)
                predicted_gestures = classifier.predict(scaled_features)

                # Display Results
                results_df = pd.DataFrame({"Predicted Gestures": predicted_gestures})
                st.dataframe(results_df, use_container_width=True)

                # Download Predictions
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=csv,
                    file_name="gesture_predictions.csv",
                    mime="text/csv"
                )

                st.success("🎯 Predictions generated successfully!")
            except Exception as err:
                st.error(f"⚠️ Prediction Error: {str(err)}")
        else:
            st.error("❌ Failed to process uploaded file. Please ensure correct format.")

# Footer
st.markdown("---")

