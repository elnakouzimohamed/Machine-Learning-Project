# utils/feature_extraction.py

import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from collections import Counter

# === Configuration ===
WINDOW_LENGTH = 100
STEP_OVERLAP_G7 = 6
STEP_NO_OVERLAP = 100
TARGET_SAMPLING_RATE = 1000

def extract_emg_features(segment):
    """
    Extract RMS, ZCR, and WL features from a signal window.
    Args:
        segment (np.array): EMG signal segment (samples x channels)
    Returns:
        np.array: Concatenated feature vector
    """
    rms = np.sqrt(np.mean(np.square(segment), axis=0))
    zcr = np.sum(np.diff(np.sign(segment), axis=0) != 0, axis=0)
    wl = np.sum(np.abs(np.diff(segment, axis=0)), axis=0)
    return np.concatenate([rms, zcr, wl])

def bandpass_filter(df):
    """
    Apply bandpass filtering (20-40Hz) to EMG signals.
    Args:
        df (pd.DataFrame): Raw EMG signals
    Returns:
        pd.DataFrame: Filtered EMG signals
    """
    nyquist = TARGET_SAMPLING_RATE / 2
    wp = 40 / nyquist
    ws = 20 / nyquist
    b, a = signal.iirdesign(wp, ws, gpass=1, gstop=60, ftype='butter')

    for col in df.columns:
        df[col] = signal.filtfilt(b, a, df[col])

    return df

def normalize_features(df):
    """
    Normalize EMG signals using standard scaling.
    Args:
        df (pd.DataFrame): EMG signals
    Returns:
        pd.DataFrame: Normalized EMG signals
    """
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

def segment_and_extract(df):
    """
    Segment EMG signals and extract features.
    Args:
        df (pd.DataFrame): Filtered and normalized EMG signals with 'class' column
    Returns:
        X, y: List of features and corresponding labels
    """
    X, y = [], []

    for gesture in sorted(df['class'].unique()):
        df_gesture = df[df['class'] == gesture]
        step = STEP_OVERLAP_G7 if gesture == 7 else STEP_NO_OVERLAP

        for start in range(0, len(df_gesture) - WINDOW_LENGTH + 1, step):
            window = df_gesture.iloc[start:start + WINDOW_LENGTH, :-1].values
            labels = df_gesture.iloc[start:start + WINDOW_LENGTH]['class'].values
            if np.all(labels == labels[0]):
                X.append(extract_emg_features(window))
                y.append(labels[0])

    return X, y

def preprocess_uploaded_data(file_path):
    """
    Full preprocessing pipeline for uploaded EMG file.
    Args:
        file_path (str): Path to uploaded .csv or .txt file
    Returns:
        pd.DataFrame or None: Processed feature DataFrame (without labels)
    """
    try:
        # Read file
        df = pd.read_csv(file_path, sep="\t")

        # Drop unwanted columns
        df = df.drop(columns=[col for col in ['time', 'label'] if col in df.columns if col in df])

        if 'class' not in df.columns:
            print("❌ Error: 'class' column missing in uploaded file.")
            return None

        # Separate features and labels
        features = df.drop(columns=['class'])
        labels = df['class']

        # Filter and normalize features
        features = bandpass_filter(features)
        features = normalize_features(features)

        # Reattach class label
        df_processed = features.copy()
        df_processed['class'] = labels

        # Segment and extract features
        X, y = segment_and_extract(df_processed)

        if X:
            feature_columns = (
                [f'RMS_{i+1}' for i in range(features.shape[1])] +
                [f'ZCR_{i+1}' for i in range(features.shape[1])] +
                [f'WL_{i+1}' for i in range(features.shape[1])]
            )

            processed_df = pd.DataFrame(X, columns=feature_columns)
            processed_df['label'] = y

            # Optionally save processed features for verification
            processed_path = os.path.join(os.path.dirname(file_path), "processed_uploaded_features.csv")
            processed_df.to_csv(processed_path, index=False)

            print(f"✅ Uploaded file processed successfully: {len(processed_df)} samples.")
            return processed_df.drop(columns=['label'])

        else:
            print("⚠️ Warning: No valid segments found in uploaded file.")
            return None

    except Exception as e:
        print(f"❌ Exception during preprocessing: {str(e)}")
        return None
