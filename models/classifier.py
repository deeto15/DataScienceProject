import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, models, losses, callbacks

# Load data
df = pd.read_csv(r"c:\Users\Kendall Eberly\Downloads\creditcard.csv")
features = df.drop("Class", axis=1)
labels = df["Class"]

# Normalize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.2, stratify=labels, random_state=42
)

# SMOTE oversampling
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# Non-fraud for autoencoder
X_train_nonfraud = X_train[y_train == 0]

# Focal loss
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow(1 - p_t, gamma) * bce
    return loss

# Classifier model
def build_classifier(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
    return model

classifier = build_classifier(X_train_bal.shape[1])
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
classifier.fit(X_train_bal, y_train_bal, epochs=200, batch_size=512, validation_split=0.1, callbacks=[early_stop], verbose=1)

# Autoencoder model
def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(16, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(encoded)
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)
    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

autoencoder = build_autoencoder(X_train.shape[1])
autoencoder.fit(X_train_nonfraud, X_train_nonfraud, epochs=200, batch_size=512, validation_split=0.1, verbose=1)

# Classifier predictions
y_pred_prob = classifier.predict(X_test).flatten()

# Autoencoder anomaly scores
reconstructions = autoencoder.predict(X_test)
recon_errors = np.mean(np.square(reconstructions - X_test), axis=1)
recon_errors_scaled = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min())

# Grid search fusion
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

# Final prediction
final_scores = best_alpha * y_pred_prob + (1 - best_alpha) * recon_errors_scaled
final_predictions = (final_scores > best_thresh).astype(int)

# Output report
print("Best alpha:", best_alpha)
print("Best threshold:", best_thresh)
print(classification_report(y_test, final_predictions))
