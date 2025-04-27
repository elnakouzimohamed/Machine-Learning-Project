# utils/prepare_dataset.py

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
UNDERSAMPLE_RATIO = 0.12         # Reduce class 0 size (only for training preparation)

def extract_features(window):
    rms = np.sqrt(np.mean(window**2, axis=0))
    zcr = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    return np.concatenate([rms, zcr, wl])

def process_emg_file(file_path):
    """Process single EMG file and extract features."""
    df = pd.read_csv(file_path, sep="\t")

    if "time" in df.columns:
        df = df.drop(columns=["time"])

    if "class" not in df.columns:
        raise ValueError(f"'class' column not found in {file_path}")

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

    # Gesture 7 with overlap
    df_g7 = df[df['class'] == 7]
    for start in range(0, len(df_g7) - WINDOW_SIZE + 1, STEP_SIZE_G7):
        window = df_g7.iloc[start:start + WINDOW_SIZE, :-1].values
        labels = df_g7.iloc[start:start + WINDOW_SIZE]['class'].values
        if len(set(labels)) == 1:
            X.append(extract_features(window))
            y.append(labels[0])

    # Other gestures (no overlap)
    df_rest = df[df['class'].isin([0, 1, 2, 3, 4, 5, 6])]
    for start in range(0, len(df_rest) - WINDOW_SIZE + 1, STEP_SIZE_OTHERS):
        window = df_rest.iloc[start:start + WINDOW_SIZE, :-1].values
        labels = df_rest.iloc[start:start + WINDOW_SIZE]['class'].values
        if len(set(labels)) == 1:
            X.append(extract_features(window))
            y.append(labels[0])

    return X, y

def summarize_labels(labels):
    counter = Counter(labels)
    print("🧮 Gesture Summary:")
    for label, count in sorted(counter.items()):
        print(f"Gesture {label}: {count} samples")

def process_all_subjects(base_dir, output_dir="subjects_csv"):
    """Process all subject folders and save CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    all_labels = []

    for subject_folder in sorted(os.listdir(base_dir)):
        subject_path = os.path.join(base_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        all_X, all_y = [], []

        for file in os.listdir(subject_path):
            if file.endswith(".txt"):
                file_path = os.path.join(subject_path, file)
                X, y = process_emg_file(file_path)
                all_X.extend(X)
                all_y.extend(y)

        if all_X:
            feature_cols = (
                [f'RMS_{i+1}' for i in range(8)] +
                [f'ZCR_{i+1}' for i in range(8)] +
                [f'WL_{i+1}' for i in range(8)]
            )
            df = pd.DataFrame(all_X, columns=feature_cols)
            df['label'] = all_y

            output_file = os.path.join(output_dir, f"subject_{subject_folder}.csv")
            df.to_csv(output_file, index=False)

            print(f"✅ Saved {output_file} with {len(df)} samples")
            all_labels.extend(all_y)
        else:
            print(f"⚠️ No valid data found for subject {subject_folder}")

    summarize_labels(all_labels)

if __name__ == "__main__":
    raw_data_dir = "EMG_data_for_gestures-master"  
    process_all_subjects(raw_data_dir)
