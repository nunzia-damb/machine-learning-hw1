from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import load_data

def hw1():
    #choose dataset 1 or dataset 2
    db = load_data.load_data("/Users/nunziadambrosio/Documents/hw1-machine-learning/dataset1.csv")
    print("dataset loaded")

    X_train, X_test, y_train, y_test = train_test_split(db[0], db[1], test_size=0.2, random_state=1)
    print("dataset split")

    #choose model
    #model = svm.SVC(C=10, kernel='linear')
    model = LogisticRegression(max_iter=1000)

    #standardization of dataset
    pipe = make_pipeline(preprocessing.StandardScaler(), model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    #scores
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_true=y_test,
                          y_pred=y_pred
                          )
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred)
    plt.show()

    #to find the best parameters:
    param_grid_svm = {'svc__C': [1, 10, 20], 'svc__kernel': ['rbf', 'linear']}
    param_grid_log = {'logisticregression__C': [1, 10, 20], 'logisticregression__penalty': ['l1'],
                      'logisticregression__solver': ['sag', 'saga']}
    #put the one you want to do grid search for
    gs = GridSearchCV(pipe, param_grid_log, cv=5, return_train_score=False, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(gs.best_params_)
    #predict values and print evaluation score and matrix
    y_pred = gs.predict(X_test)  # calls predict with best found parameters
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_true=y_test,
                          y_pred=y_pred
                          )
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred)
    plt.show()

if __name__ == '__main__':
    hw1()