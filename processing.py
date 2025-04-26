# processing.py

import os
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from collections import Counter

# === CONFIGURATION ===
WINDOW_SIZE = 100
STEP_SIZE_G7 = 6                # 94% overlap for gesture 7
STEP_SIZE_OTHERS = 100          # No overlap for gestures 0–6
FS = 1000
UNDERSAMPLE_RATIO = 0.1          # Reduce class 0 size

# === FEATURE EXTRACTION ===
def extract_features(window):
    rms = np.sqrt(np.mean(window**2, axis=0))
    zcr = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    return np.concatenate([rms, zcr, wl])

# === PROCESS ONE FILE ===
def process_emg_file_selective_overlap(file_path):
    df = pd.read_csv(file_path, sep="\t")

    # Drop 'time' if it exists
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    # Drop 'label' if it exists
    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Undersample class 0
    zeros = df[df['class'] == 0]
    nonzeros = df[df['class'] != 0]
    if not zeros.empty:
        zeros = zeros.sample(frac=UNDERSAMPLE_RATIO, random_state=42)
    df = pd.concat([zeros, nonzeros]).sort_index().reset_index(drop=True)

    # Bandpass filter (20–40 Hz)
    wp = 40 / (FS / 2)
    ws = 20 / (FS / 2)
    b, a = signal.iirdesign(wp, ws, gpass=1, gstop=60, ftype='butter')
    for ch in df.columns[:-1]:
        df[ch] = signal.filtfilt(b, a, df[ch])

    # Normalize
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    X, y = [], []

    # Process gesture 7 with 50% overlap
    df_g7 = df[df['class'] == 7]
    for start in range(0, len(df_g7) - WINDOW_SIZE + 1, STEP_SIZE_G7):
        window = df_g7.iloc[start:start + WINDOW_SIZE, :-1].values
        labels = df_g7.iloc[start:start + WINDOW_SIZE]['class'].values
        if len(set(labels)) == 1:
            X.append(extract_features(window))
            y.append(labels[0])

    # Process other gestures (0–6) with NO overlap
    df_rest = df[df['class'].isin([0, 1, 2, 3, 4, 5, 6])]
    for start in range(0, len(df_rest) - WINDOW_SIZE + 1, STEP_SIZE_OTHERS):
        window = df_rest.iloc[start:start + WINDOW_SIZE, :-1].values
        labels = df_rest.iloc[start:start + WINDOW_SIZE]['class'].values
        if len(set(labels)) == 1:
            X.append(extract_features(window))
            y.append(labels[0])

    return X, y


# === Print samples per gesture (optional) ===
def print_gesture_summary(label_list):
    counter = Counter(label_list)
    print("\n🧮 Overall Sample Count per Gesture:")
    for gesture in sorted(counter):
        print(f"Gesture {int(gesture)}: {counter[gesture]} samples")

# === Main function to process 'uploads' folder ===
def process_uploads_folder(upload_dir="uploads"):
    files = [f for f in os.listdir(upload_dir) if f.endswith(".txt")]
    if not files:
        print("❌ No .txt file found in uploads folder.")
        return None

    file_path = os.path.join(upload_dir, files[0])
    print(f"🔍 Processing file: {file_path}")

    X, y = process_emg_file_selective_overlap(file_path)

    if X:
        feature_cols = [f'RMS_{i+1}' for i in range(8)] + \
                       [f'ZCR_{i+1}' for i in range(8)] + \
                       [f'WL_{i+1}' for i in range(8)]

        df = pd.DataFrame(X, columns=feature_cols)
        df['label'] = y

        print_gesture_summary(y)

        # Save processed file as "processed.csv"
        processed_path = os.path.join(upload_dir, "processed.csv")
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed data saved to {processed_path}")

        return df.drop(columns=['label'])  # Return features only for prediction
    else:
        print("⚠️ No valid data found.")
        return None
