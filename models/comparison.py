import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import models
from sklearn.metrics import classification_report, confusion_matrix

def train_and_stack_creditcard(df_path):
    df = pd.read_csv(df_path)
    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    hgb_param = {"max_iter":[100,300,500],"max_depth":[10,15],"learning_rate":[0.1,0.5,1.0]}
    hgb_search = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=42), hgb_param, n_iter=5, cv=2, scoring="f1", n_jobs=-1, random_state=42)
    hgb_search.fit(X_train, y_train)
    hgb_best = hgb_search.best_estimator_

    xgb_param = {"n_estimators":[100,300,500],"max_depth":[6,10],"learning_rate":[0.01,0.1],"subsample":[0.8,1.0]}
    xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric="logloss", tree_method="hist"), xgb_param, n_iter=5, cv=2, scoring="f1", n_jobs=-1, random_state=42)
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_

    autoencoder = models.load_model("autoencoder.keras")
    ae_err_test = np.mean((autoencoder.predict(X_test) - X_test)**2, axis=1)
    ae_err_test_scaled = (ae_err_test - ae_err_test.min())/(ae_err_test.max()-ae_err_test.min())

    hgb_probs = hgb_best.predict_proba(X_test)[:,1]
    xgb_probs = xgb_best.predict_proba(X_test)[:,1]

    meta_X = np.column_stack([hgb_probs, xgb_probs, ae_err_test_scaled])
    meta = LogisticRegression().fit(meta_X, y_test)
    final_preds = meta.predict(meta_X)

    print(classification_report(y_test, final_preds))
    print(confusion_matrix(y_test, final_preds))

# example usage:
train_and_stack_creditcard(r"C:\Users\Kendall Eberly\Downloads\creditcard.csv")
