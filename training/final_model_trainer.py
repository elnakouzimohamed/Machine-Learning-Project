# training/final_model_trainer.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# === CONFIGURATION ===
DATA_DIR = "./subjects_csv"
LOS0_RESULTS_PATH = "./rf_fast_results/all_loso_results.csv"
MODEL_SAVE_PATH = "./saved_models/gesture_model.pkl"

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def find_best_model():
    """Read LOSO results and find the best model by F1 Score."""
    results_df = pd.read_csv(LOS0_RESULTS_PATH)

    # Fill NaNs temporarily for grouping
    results_df['n_estimators'] = results_df['n_estimators'].fillna(-1)
    results_df['C'] = results_df['C'].fillna(-1)
    results_df['kernel'] = results_df['kernel'].fillna("none")

    # Group and find best
    grouped = results_df.groupby(['Model', 'n_estimators', 'C', 'kernel']).agg({
        'F1 Score': 'mean'
    }).reset_index()

    if grouped.empty:
        raise ValueError("No valid model results found. Please ensure LOSO evaluation was completed properly.")

    best_idx = grouped['F1 Score'].idxmax()
    best_row = grouped.iloc[best_idx]

    model_type = best_row['Model']
    n_estimators = None if best_row['n_estimators'] == -1 else int(best_row['n_estimators'])
    C = None if best_row['C'] == -1 else float(best_row['C'])
    kernel = None if best_row['kernel'] == "none" else best_row['kernel']

    print(f"🏆 Best Model Selected: {model_type}, Parameters: n_estimators={n_estimators}, C={C}, kernel={kernel}")
    return model_type, n_estimators, C, kernel

def retrain_final_model(model_type, n_estimators, C, kernel):
    """Retrain the selected best model on all subject data."""
    subject_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    all_data = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f)) for f in subject_files], ignore_index=True)

    X = all_data.drop(columns=['label'])
    y = all_data['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_type == "SVM":
        model = SVC(C=C, kernel=kernel, probability=False, random_state=42)
    else:
        raise ValueError("Unknown model type!")

    model.fit(X_scaled, y)

    # Save both scaler and model
    joblib.dump((scaler, model), MODEL_SAVE_PATH)
    print(f"✅ Final model saved at {MODEL_SAVE_PATH}")

def main():
    model_type, n_estimators, C, kernel = find_best_model()
    retrain_final_model(model_type, n_estimators, C, kernel)

if __name__ == "__main__":
    main()
