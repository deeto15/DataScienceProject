import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


# paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
data_path = os.path.join(downloads_folder, "creditcard.csv")


# Load models
classifier = load_model(os.path.join(model_dir, "classifier_nn.keras"))
autoencoder = load_model(os.path.join(model_dir, "autoencoder.keras"))
rf_model = joblib.load(os.path.join(model_dir, "randomforest_model.pkl"))
dt_model = joblib.load(os.path.join(model_dir, "decisiontree_model.pkl"))
lr_model = joblib.load(os.path.join(model_dir, "logistic_regression_model.pkl"))
