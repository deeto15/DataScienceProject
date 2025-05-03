import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


# paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
data_path = os.path.join(downloads_folder, "creditcard.csv")

# Load data and preprocess
df = pd.read_csv(data_path)
x = df.drop("Class", axis=1).values
y = df["Class"].values

scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)

# we will not train/test split. all data will be used for testing
X_test = X_scaled
y_test = y

# Load models
# classifier = load_model(os.path.join(model_dir, "classifier_nn.keras"))
# autoencoder = load_model(os.path.join(model_dir, "autoencoder.keras"))
rf_model = joblib.load(os.path.join(model_dir, "randomforest_model.pkl"))
dt_model = joblib.load(os.path.join(model_dir, "decisiontree_model.pkl"))
lr_model = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))

# Generate prediction probabilities
model_scores = {
    "Random Forest": rf_model.predict_proba(X_scaled)[:, 1],
    "Decision Tree": dt_model.predict_proba(X_scaled)[:, 1],
    "Logistic Regression": lr_model.predict_proba(X_scaled)[:, 1],
}

# ROC Curve
plt.subplot(1, 2, 1)
for name, scores in model_scores.items():
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 2, 2)
for name, scores in model_scores.items():
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    plt.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()
