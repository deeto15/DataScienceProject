import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        bce = losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow(1 - p_t, gamma) * bce

    return loss


def build_classifier(input_dim):
    model = models.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(input_dim,)),
            layers.Dropout(0.4),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss=focal_loss(), metrics=["accuracy"])
    return model


def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation="relu")(input_layer)
    encoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(32, activation="relu")(encoded)
    output_layer = layers.Dense(input_dim, activation="linear")(decoded)
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def prepare_data():
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_folder, "creditcard.csv")
    df = pd.read_csv(file_path)
    features = df.drop("Class", axis=1)
    labels = df["Class"]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, stratify=labels, random_state=42
    )
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    X_train_nonfraud = X_train[y_train == 0]
    return X_train_bal, y_train_bal, X_train_nonfraud, X_test, y_test, features.shape[1]


def train_and_save_models():
    X_train_bal, y_train_bal, X_train_nonfraud, _, _, input_dim = prepare_data()
    classifier = build_classifier(input_dim)
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    classifier.fit(
        X_train_bal,
        y_train_bal,
        epochs=100,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(
        X_train_nonfraud,
        X_train_nonfraud,
        epochs=100,
        batch_size=512,
        validation_split=0.1,
        verbose=1,
    )
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )
    classifier.save(os.path.join(output_dir, "classifier_nn.keras"), include_optimizer=False)
    autoencoder.save(os.path.join(output_dir, "autoencoder.keras"), include_optimizer=False)
    print("Models saved successfully.")


def evaluate_models():
    _, _, X_train_nonfraud, X_test, y_test, input_dim = prepare_data()
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    )
    classifier = models.load_model(
        os.path.join(output_dir, "classifier_nn.keras"),
        custom_objects={"loss": focal_loss()},
    )
    autoencoder = models.load_model(os.path.join(output_dir, "autoencoder.keras"))
    y_pred_prob = classifier.predict(X_test).flatten()
    reconstructions = autoencoder.predict(X_test)
    recon_errors = np.mean(np.square(reconstructions - X_test), axis=1)
    recon_errors_scaled = (recon_errors - recon_errors.min()) / (
        recon_errors.max() - recon_errors.min()
    )
    best_f1 = 0
    best_alpha = 0
    best_thresh = 0
    alphas = np.linspace(0.1, 0.9, 9)
    thresholds = np.linspace(0.1, 0.9, 17)
    for alpha in alphas:
        fusion_score = alpha * y_pred_prob + (1 - alpha) * recon_errors_scaled
        for t in thresholds:
            preds = (fusion_score > t).astype(int)
            f1 = f1_score(y_test, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_alpha = alpha
                best_thresh = t
    final_scores = best_alpha * y_pred_prob + (1 - best_alpha) * recon_errors_scaled
    final_predictions = (final_scores > best_thresh).astype(int)
    print("Best alpha:", best_alpha)
    print("Best threshold:", best_thresh)
    print(classification_report(y_test, final_predictions))
    print("Confusion Matrix:")
    print(
        "["
        + "\n ".join(
            [str(row.tolist()) for row in confusion_matrix(y_test, final_predictions)]
        )
        + "]"
    )


def doesExist():
    model_paths = ["classifier_nn.keras", "autoencoder.keras"]
    if not all(
        os.path.exists(
            os.path.join(
                os.path.abspath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
                ),
                p,
            )
        )
        for p in model_paths
    ):
        train_and_save_models()
    evaluate_models()
