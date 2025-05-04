import pre_processing as pp
from classifier import doesExist
from sklearn.model_selection import train_test_split


def main():
    raw_data = pp.load("creditcard.csv")

    # Full test set from original data
    full_X = raw_data.drop(["Class"], axis=1)
    full_y = raw_data["Class"]
    _, X_full_test, _, y_full_test = train_test_split(
        full_X, full_y, stratify=full_y, test_size=0.2, random_state=42
    )

    print("Random Forest")
    rf_model, _, _ = pp.training_random_forest(raw_data)
    pp.save_model(rf_model, "randomforest_model.pkl")
    pp.test(rf_model, X_full_test, y_full_test, None)

    print("Logistic Regression")
    lr_model, _,_ = pp.training_logistic_regression(raw_data)
    pp.save_model(lr_model, "logistic_regression_model.pkl")
    pp.test(lr_model, X_full_test, y_full_test, None)

    print("Autoencoder/Predictor")
    doesExist()


main()
