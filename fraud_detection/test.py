import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def load_data():
    X,y = make_classification(n_samples=10000)
    return X,y
def load_cancer_data():
    X,y = load_breast_cancer(return_X_y=True)
    return X,y
def prepare_data(X,y):
    train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.2)
    return train_x, test_x, train_y, test_y
def do_train(train_x, test_x, train_y, test_y):
    model = LogisticRegression(max_iter=1000)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    error = classification_report(test_y, preds)
    print(error)
if __name__ == '__main__':
    X,y= load_cancer_data()
    train_x, test_x, train_y, test_y = prepare_data(X,y)
    do_train(train_x, test_x, train_y, test_y)
    print(X)