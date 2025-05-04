import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle  # used to save the models
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


# load data
def load(filename):
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_folder, filename)
    return pd.read_csv(file_path)

# train the model using randomforestclassifier
def training_random_forest(data):
    X = data.drop(["Class"], axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=1000,  # more trees = better ensemble
        max_depth=12,  # allow deeper trees for capturing more structure
        min_samples_split=10,  # reduce splitting on noise
        min_samples_leaf=4,  # enforce smoother trees
        max_features="sqrt",  # diversify trees by feature subsampling
        class_weight="balanced",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test


# train the model using logisticregression
def training_logistic_regression(data):
    X = data.drop(["Class"], axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def tuned_xgboost(X_train, y_train):
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'scale_pos_weight': [scale]
    }
    xgb = RandomizedSearchCV(XGBClassifier(n_estimators=500, eval_metric='logloss', random_state=42),
                             param_dist, n_iter=10, cv=3, scoring='f1', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    return xgb.best_estimator_

# save model to disk
def save_model(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


# test the model
def test(model, X_test, y_test, w_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, sample_weight=w_test)
    cm = confusion_matrix(y_test, predictions, sample_weight=w_test)
    print(report)
    print("Confusion Matrix:")
    for row in cm.astype(int):
        print(" ", row.tolist())
