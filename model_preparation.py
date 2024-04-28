import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df_train = pd.read_csv('train.csv')

y = df_train[['Хороший_день']]
X = df_train.drop('Хороший_день', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=42)         

log_regression = LogisticRegression()
log_regression.fit(X_train,y_train.values.ravel())

import pickle
# save
with open('model.pkl','wb') as f:
    pickle.dump(log_regression,f)

