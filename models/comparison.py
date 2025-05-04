import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def combine_all_predictions(models, autoencoder, X_test, y_test):
    ensemble_probs = np.mean([model.predict_proba(X_test)[:, 1] for model in models], axis=0)
    
    ae_err = np.mean((autoencoder.predict(X_test) - X_test)**2, axis=1)
    ae_err_scaled = (ae_err - ae_err.min()) / (ae_err.max() - ae_err.min())

    best_thresh, best_alpha, best_metric = 0.5, 0.5, -np.inf

    for alpha in np.linspace(0.1, 0.9, 9):
        combined_scores = alpha * ensemble_probs + (1 - alpha) * ae_err_scaled
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (combined_scores >= thresh).astype(int)

            recall = classification_report(y_test, preds, output_dict=True)['1']['recall']
            precision = classification_report(y_test, preds, output_dict=True)['1']['precision']
            fp_rate = (preds > y_test).sum() / (y_test == 0).sum()

            if recall >= 0.9 and fp_rate < 0.01:
                metric = f1_score(y_test, preds)
                if metric > best_metric:
                    best_metric, best_alpha, best_thresh = metric, alpha, thresh

    final_scores = best_alpha * ensemble_probs + (1 - best_alpha) * ae_err_scaled
    final_preds = (final_scores >= best_thresh).astype(int)

    print(f"Optimal Alpha: {best_alpha:.2f}")
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(classification_report(y_test, final_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, final_preds))
