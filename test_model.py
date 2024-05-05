import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_creation import X_test, y_test
from sklearn import metrics
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,  confusion_matrix, roc_curve, accuracy_score
# load
with open('regress_iris.pkl', 'rb') as f:
    regress_iris = pickle.load(f)
    
# Evaluate the model
y_pred = regress_iris.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
