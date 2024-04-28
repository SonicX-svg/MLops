import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_preparation import X_test, y_test
from sklearn import metrics
import pickle
# load
with open('model.pkl', 'rb') as f:
    log_regression = pickle.load(f)
    
y_pred = log_regression.predict(X_test)
#print(y_pred)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
acuracy_score = metrics.accuracy_score(y_test, y_pred)
print('cnf_matrix',cnf_matrix, '\n', 'acuracy_score',acuracy_score)
