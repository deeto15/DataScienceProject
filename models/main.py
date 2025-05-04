import pre_processing as pp
from classifier import doesExist, build_autoencoder, focal_loss, prepare_data
from comparison import combine_all_predictions
from sklearn.model_selection import train_test_split
import tensorflow as tf

def main():
    data = pp.load("creditcard.csv")
    X = data.drop("Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    X_train_smote, y_train_smote = pp.apply_smote(X_train, y_train)

    rf_model = pp.training_random_forest(X_train_smote, y_train_smote)
    lr_model = pp.training_logistic_regression(X_train_smote, y_train_smote)
    hgb_model = pp.training_hist_gradient(X_train_smote, y_train_smote)
    xgb_model = pp.training_xgboost(X_train_smote, y_train_smote)

    pp.save_model(rf_model, "randomforest_model.pkl")
    pp.save_model(lr_model, "logistic_regression_model.pkl")
    pp.save_model(hgb_model, "histgradient_model.pkl")
    pp.save_model(xgb_model, "xgboost_model.pkl")

    print("Random Forest Performance:")
    pp.test(rf_model, X_test, y_test)

    print("Logistic Regression Performance:")
    pp.test(lr_model, X_test, y_test)

    print("HistGradientBoosting Performance:")
    pp.test(hgb_model, X_test, y_test)

    print("XGBoost Performance:")
    pp.test(xgb_model, X_test, y_test)

    doesExist()

    _, _, X_train_nonfraud, _, _, input_dim = prepare_data()
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(X_train_nonfraud, X_train_nonfraud, epochs=50, batch_size=512, verbose=0)
    autoencoder.save("autoencoder.keras", include_optimizer=False)

    autoencoder = tf.keras.models.load_model("autoencoder.keras", compile=False)

    print("Combined Models with Autoencoder (Optimized Recall):")
    combine_all_predictions([rf_model, lr_model, hgb_model, xgb_model], autoencoder, X_test, y_test)

main()
