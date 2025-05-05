import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
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

# Prepare scaled version (for Logistic Regression and neural models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Unscaled for tree-based models
X_unscaled_df = X.copy()

# Load models
rf_model = joblib.load(os.path.join(model_dir, "randomforest_model.pkl"))
lr_model = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))
hist_model = joblib.load(os.path.join(model_dir, "histgradient_model.pkl"))
xgb_model = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
nn_model = tf.keras.models.load_model(os.path.join(model_dir, "classifier_nn.keras"), compile=False)
ae_model = tf.keras.models.load_model(os.path.join(model_dir, "autoencoder.keras"), compile=False)

# Dataset dictionary (only full dataset now)
datasets = {
    "Full Dataset": {
        "scaled": X_scaled_df,
        "unscaled": X_unscaled_df,
        "y": y
    }
}

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Model Comparison: Full Dataset", fontsize=16)

for row_idx, (label, data) in enumerate(datasets.items()):
    X_scaled = data["scaled"]
    X_unscaled = data["unscaled"]
    y_eval = data["y"]

    # Autoencoder scores (reconstruction error)
    ae_reconstructed = ae_model.predict(X_scaled, verbose=0)
    ae_scores = np.mean(np.square(X_scaled - ae_reconstructed), axis=1)

    # Predict_proba or equivalent for all models
    model_scores = {
        "Random Forest": rf_model.predict_proba(X_unscaled)[:, 1],
        "HistGradient": hist_model.predict_proba(X_unscaled)[:, 1],
        "Logistic Regression": lr_model.predict_proba(X_scaled)[:, 1],
        "XGBoost": xgb_model.predict_proba(X_unscaled)[:, 1],
        "Neural Net Classifier": nn_model.predict(X_scaled, verbose=0).flatten(),
        "Autoencoder": ae_scores
    }

    # Ensemble prediction: average of all model outputs
    ensemble_scores = np.mean(np.column_stack(list(model_scores.values())), axis=1)
    model_scores["Ensemble Average"] = ensemble_scores

    # ROC Curve
    ax_roc = axes[0]
    for name, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_eval, scores)
        auc = roc_auc_score(y_eval, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set_title(f"ROC - {label}")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()

    # Precision-Recall Curve
    ax_pr = axes[1]
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
