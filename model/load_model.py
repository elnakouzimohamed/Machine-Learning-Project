# model/load_model.py

import joblib
import os

MODEL_PATH = "saved_models/gesture_model.pkl"

def get_trained_model():
    """
    Load the pre-trained scaler and model from disk.
    Returns:
        scaler: Preprocessing scaler
        model: Trained machine learning model
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    
    scaler, model = joblib.load(MODEL_PATH)
    return scaler, model
