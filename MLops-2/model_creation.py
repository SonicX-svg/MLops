from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
df_train = pd.read_csv('train.csv')

y = df_train[['target']]
X = df_train.drop('target', axis=1)

# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости результата
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train.values.ravel())


import pickle
with open('neigh_model.pkl','wb') as f:
    pickle.dump(neigh,f)
