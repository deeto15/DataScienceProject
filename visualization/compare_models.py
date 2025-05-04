import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
data_path = os.path.join(downloads_folder, "creditcard.csv")

# Load and preprocess data
df = pd.read_csv(data_path)
X = df.drop("Class", axis=1)
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Downsample to balance fraud and non-fraud
fraud_df = df[df["Class"] == 1]
nonfraud_df = df[df["Class"] == 0].sample(n=len(fraud_df), random_state=42)
downsampled_df = pd.concat([fraud_df, nonfraud_df]).sample(frac=1, random_state=42)  # shuffle

X_down = downsampled_df.drop("Class", axis=1)
y_down = downsampled_df["Class"].values
X_down_scaled = scaler.transform(X_down)
X_down_df = pd.DataFrame(X_down_scaled, columns=X.columns)

# Load models
rf_model = joblib.load(os.path.join(model_dir, "randomforest_model.pkl"))
dt_model = joblib.load(os.path.join(model_dir, "decisiontree_model.pkl"))
lr_model = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))

# Store scores for both datasets
datasets = {
    "Full Dataset": (X_scaled_df, y),
    "Downsampled Dataset": (X_down_df, y_down)
}

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Comparison: Full vs. Downsampled Dataset", fontsize=16)

for row_idx, (label, (X_eval, y_eval)) in enumerate(datasets.items()):
    # Prediction scores
    model_scores = {
        "Random Forest": rf_model.predict_proba(X_eval)[:, 1],
        "Decision Tree": dt_model.predict_proba(X_eval)[:, 1],
        "Logistic Regression": lr_model.predict_proba(X_eval)[:, 1],
    }

    # ROC Curve
    ax_roc = axes[row_idx, 0]
    for name, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_eval, scores)
        auc = roc_auc_score(y_eval, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_title(f"ROC - {label}")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()

    # PR Curve
    ax_pr = axes[row_idx, 1]
    for name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_eval, scores)
        ap = average_precision_score(y_eval, scores)
        ax_pr.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")
    ax_pr.set_title(f"Precision-Recall - {label}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
