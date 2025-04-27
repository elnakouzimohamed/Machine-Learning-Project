# training/model_selection_plot.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
RESULTS_DIR = "./rf_fast_results"
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_random_forest_results():
    """Plot Random Forest: n_estimators vs Accuracy."""
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("rf_") and f.endswith(".csv")]
    data = []

    for file in files:
        df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        n_estimators = int(df['n_estimators'].iloc[0])
        avg_accuracy = df['Accuracy'].mean()
        data.append((n_estimators, avg_accuracy))

    data.sort(key=lambda x: x[0])  # Sort by number of trees

    n_estimators_list, avg_accuracies = zip(*data)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, avg_accuracies, marker='o')
    plt.title('Random Forest - n_estimators vs Average Accuracy')
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('Average Accuracy')
    plt.grid(True)
    plt.xticks(n_estimators_list)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "rf_n_estimators_vs_accuracy.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"✅ Random Forest plot saved at {save_path}")

def plot_svm_results():
    """Plot SVM: C values vs Accuracy for each kernel."""
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("svm_") and f.endswith(".csv")]
    results = []

    for file in files:
        df = pd.read_csv(os.path.join(RESULTS_DIR, file))
        kernel = df['kernel'].iloc[0]
        C = df['C'].iloc[0]
        avg_accuracy = df['Accuracy'].mean()
        results.append((kernel, C, avg_accuracy))

    # Split by kernel
    kernels = set([r[0] for r in results])
    for kernel in kernels:
        kernel_data = [(C, acc) for k, C, acc in results if k == kernel]
        kernel_data.sort(key=lambda x: x[0])  # sort by C

        C_list, accuracy_list = zip(*kernel_data)

        plt.figure(figsize=(10, 6))
        plt.plot(C_list, accuracy_list, marker='o')
        plt.title(f'SVM ({kernel} kernel) - C vs Average Accuracy')
        plt.xlabel('C value')
        plt.ylabel('Average Accuracy')
        plt.grid(True)
        plt.xticks(C_list)
        plt.tight_layout()
        save_path = os.path.join(PLOT_DIR, f"svm_{kernel}_c_vs_accuracy.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

        print(f"✅ SVM ({kernel}) plot saved at {save_path}")

def main():
    plot_random_forest_results()
    plot_svm_results()

if __name__ == "__main__":
    main()
