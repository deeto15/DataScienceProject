import pre_processing as pp


def main():
    raw_data = pp.load("creditcard.csv")
    downsampled = pp.downsampling(raw_data)
    upweighted = pp.upweight(downsampled)

    print("Random Forest")  
    randomforest_model, X_test, y_test, w_test = pp.training_random_forest(upweighted)
    pp.test(randomforest_model, X_test, y_test, w_test)

    print("Decision Tree")
    decisiontree_model, X_test, y_test, w_test = pp.training_decision_tree(upweighted)
    pp.test(decisiontree_model, X_test, y_test, w_test)

    print("Logistic Regression")
    LogisticRegression_model, X_test, y_test, w_test = pp.training_logistic_regression(upweighted)
    pp.test(LogisticRegression_model, X_test, y_test, w_test)
    
    
main()