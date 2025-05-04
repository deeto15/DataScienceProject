import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load(filename):
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    return pd.read_csv(os.path.join(downloads_folder, filename))

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

def training_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=1000, max_depth=12, min_samples_split=10,
                                   min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def training_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def training_hist_gradient(X_train, y_train):
    params = {"max_iter": [100, 300], "max_depth": [10, 15], "learning_rate": [0.1, 0.5]}
    search = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=42), params,
                                n_iter=5, cv=3, scoring="recall", n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    return search.best_estimator_

def training_xgboost(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        'max_depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9], 'scale_pos_weight': [scale_pos_weight]}
    search = RandomizedSearchCV(XGBClassifier(n_estimators=500, eval_metric='logloss', random_state=42),
                                params, n_iter=5, cv=3, scoring='recall', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_

def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def test(model, X_test, y_test):
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
