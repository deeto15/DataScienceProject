import pre_processing as pp
import subprocess

def main():
    raw_data = pp.load("creditcard.csv")
    downsampled = pp.downsampling(raw_data)
    upweighted = pp.upweight(downsampled)

    print("Random Forest")  
    randomforest_model, X_test, y_test, w_test = pp.training_random_forest(upweighted)
    pp.save_model(randomforest_model, "randomforest_model.pkl")
    pp.test(randomforest_model, X_test, y_test, w_test)

    print("Decision Tree")
    decisiontree_model, X_test, y_test, w_test = pp.training_decision_tree(upweighted)
    pp.save_model(decisiontree_model, "decisiontree_model.pkl")
    pp.test(decisiontree_model, X_test, y_test, w_test)

    print("Logistic Regression")
    LogisticRegression_model, X_test, y_test, w_test = pp.training_logistic_regression(upweighted)
    pp.save_model(LogisticRegression_model, "logistic_regression_model.pkl")
    pp.test(LogisticRegression_model, X_test, y_test, w_test)
    
    subprocess.run(["streamlit", "run", "app.py"])
    
main()