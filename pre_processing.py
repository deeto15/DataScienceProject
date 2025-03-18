import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#load data
def load(filename):
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_folder, filename)
    return pd.read_csv(file_path)
    
#reduce the number of negative examples to be the same as positives since fraud is so rare
def downsampling(data):
    positive = data[data['Class'] == 1]
    negative = data[data['Class'] == 0]
    downsample_negative = resample(negative, replace=False, n_samples=len(positive), random_state=42)
    return pd.concat([positive, downsample_negative])

#make it so those negative examples are weighted by a factor of which they were downsampled
def upweight(data):
    data['Weight'] = data['Class'].apply(lambda x: 600 if x == 0 else 1)
    return data

#train the model using randomforestclassifier
def training(data):
    X = data.drop(['Class', 'Weight'], axis=1)
    y = data['Class']
    weight = data['Weight']
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weight, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train, sample_weight=w_train)
    return model, X_test, y_test, w_test

#test the model
def test(model, X_test, y_test, w_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, sample_weight=w_test)
    print(report)

def main():
    raw_data = load("creditcard.csv")
    downsampled = downsampling(raw_data)
    upweighted = upweight(downsampled)
    model, X_test, y_test, w_test = training(upweighted)
    test(model, X_test, y_test, w_test)
    
main()