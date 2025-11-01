import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Use Agg if TkAgg doesn't work
import matplotlib.pyplot as plt

# Define models and paths
models = ['gcn', 'gat', 'sage', 'gin']
colors = {'gcn': 'blue', 'gat': 'orange', 'sage': 'green', 'gin': 'red'}

# Paths
log_dir = "logs"

# ---------- LOSS vs EPOCH (Log scale) ----------
plt.figure(figsize=(10, 6))
for model in models:
    file_path = os.path.join(log_dir, f"{model}_metrics.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)
    # Avoid log(0) by adding a small epsilon
    plt.plot(df['Epoch'], df['Loss'] + 1e-8, label=model.upper(), color=colors[model])

plt.yscale('log')  # Log scale for better visibility
plt.title("Loss vs Epochs (Log Scale)")
plt.xlabel("Epochs")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("logs/loss_vs_epochs_log_scaled.png")
plt.show()

# ---------- ACCURACY vs EPOCH (Smoothed, Zoomed) ----------
plt.figure(figsize=(10, 6))
for model in models:
    file_path = os.path.join(log_dir, f"{model}_metrics.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # Optional: smooth accuracy with rolling average (window=5)
    df['Smoothed Accuracy'] = df['Test Accuracy'].rolling(window=5, min_periods=1).mean()

    plt.plot(df['Epoch'], df['Smoothed Accuracy'], label=model.upper(), color=colors[model])

plt.title("Test Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.ylim(0.6, 0.9)  # Zoom into the relevant accuracy range
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/accuracy_vs_epochs.png")
plt.show()
