import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_creation import X_test, y_test
from sklearn import metrics
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix 

# load
with open('neigh_model.pkl', 'rb') as f:
    neigh_model = pickle.load(f)
    
y_pred = neigh_model.predict(X_test)
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))

