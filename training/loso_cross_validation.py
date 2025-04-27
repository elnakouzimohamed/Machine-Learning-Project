# training/loso_cross_validation.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from metrics import compute_metrics

# === CONFIGURATION ===
DATA_DIR = "./subjects_csv"
RF_ESTIMATORS = [50, 100, 150, 200]  # Random Forest trees to test
SVM_CS = [0.1, 1, 10]                # SVM regularization strength
SVM_KERNELS = ['linear', 'rbf']       # SVM kernels to test
N_JOBS = -1  # Use all CPU cores
RESULTS_DIR = "./rf_fast_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_subjects():
    """Load all subject CSVs into a list."""
    subject_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    subjects = [(f, pd.read_csv(os.path.join(DATA_DIR, f))) for f in subject_files]
    return subjects

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate a model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred)

def loso_evaluate(subjects, model_builder, model_name, param_info):
    """LOSO Cross-validation evaluation."""
    results = []

    for i, (test_file, test_df) in enumerate(subjects):
        train_df = pd.concat([df for j, (_, df) in enumerate(subjects) if j != i], ignore_index=True)

        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']
        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_builder()

        metrics_result = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
        metrics_result.update({
            'Subject': test_file,
            'Model': model_name,
            **param_info
        })

        results.append(metrics_result)

        print(f"✅ [{i+1}/{len(subjects)}] {model_name} ({param_info}) done.")

    return results

def run_loso_cross_validation():
    """Run full LOSO cross-validation for all models and save results."""
    subjects = load_subjects()

    all_results = []

    # === Random Forest ===
    for n_estimators in RF_ESTIMATORS:
        print(f"🌲 Running Random Forest with {n_estimators} trees...")
        results = loso_evaluate(
            subjects,
            model_builder=lambda: RandomForestClassifier(n_estimators=n_estimators, random_state=42),
            model_name='RandomForest',
            param_info={'n_estimators': n_estimators}
        )
        all_results.extend(results)
        pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/rf_{n_estimators}_trees.csv", index=False)

    # === SVM ===
    for C in SVM_CS:
        for kernel in SVM_KERNELS:
            print(f"⚡ Running SVM with C={C}, kernel={kernel}...")
            results = loso_evaluate(
                subjects,
                model_builder=lambda: SVC(C=C, kernel=kernel, probability=False, random_state=42),
                model_name='SVM',
                param_info={'C': C, 'kernel': kernel}
            )
            all_results.extend(results)
            pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/svm_c{C}_kernel{kernel}.csv", index=False)

    print("\n✅ All LOSO evaluations completed!")

    # Save merged results
    full_results_df = pd.DataFrame(all_results)
    full_results_df.to_csv(f"{RESULTS_DIR}/all_loso_results.csv", index=False)
    print(f"📁 Full LOSO results saved to {RESULTS_DIR}/all_loso_results.csv")

if __name__ == "__main__":
    run_loso_cross_validation()
