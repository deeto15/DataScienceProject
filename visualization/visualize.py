import os
import sys
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd

# Add parent directory to sys.path to import preprocessing
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
sys.path.append(PARENT_DIR)

import models.pre_processing as pp

# Define an output folder within the same directory
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------create visuals for data ------------------------------------


# count the number of fraud and non-fraud cases
def plot_fraud_counts(df, title):
    fraud_counts = (
        df["Class"].value_counts().rename({0: "Not Fraud", 1: "Fraud"}).reset_index()
    )
    fraud_counts.columns = ["Class", "Count"]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=fraud_counts,
        x="Class",
        y="Count",
        hue="Class",
        palette="viridis",
        legend=False,
    )
    plt.title("Fraud vs Non-Fraud Cases")
    plt.xlabel("Class")
    plt.ylabel("Count")

    # Add labels on top of each bar using actual bar positions
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
            label_type="edge",
            padding=3,
            fontsize=10,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, title))
    plt.close()


# -------------------------------------------------------------------------------


# -------------------create graph of feature importance---------------------
def plot_feature_importance(model_path, data, title):
    # Load the model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Get feature importance
    feature_importance = model.feature_importances_

    # Create a DataFrame for feature importance
    feature_names = data.columns[:-1]  # Exclude the target variable
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        hue="Feature",
        palette="viridis",
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, title))
    plt.close()


if __name__ == "__main__":
    # load the data
    df = pp.load("creditcard.csv")

    # plot the raw data fraud counts (before processing)
    plot_fraud_counts(df, "fraud_counts_raw.png")
    print(
        f"Fraud counts (raw) plot saved to {os.path.join(OUTPUT_DIR, 'fraud_counts_raw.png')}"
    )

    # downdsample the data
    downsampled_df = pp.downsampling(df)
    # plot the downsampled data fraud counts (after processing)
    plot_fraud_counts(downsampled_df, "fraud_counts_downsampled.png")
    print(
        f"Fraud counts (downsampled) plot saved to {os.path.join(OUTPUT_DIR, 'fraud_counts_downsampled.png')}"
    )

    # plot feature importance for random forest model
    rf_model_path = os.path.join(PARENT_DIR, "randomforest_model.pkl")
    plot_feature_importance(rf_model_path, downsampled_df, "feature_importance_rf.png")
    print(
        f"Feature importance (Random Forest) plot saved to {os.path.join(OUTPUT_DIR, 'feature_importance_rf.png')}"
    )
