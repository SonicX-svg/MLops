import pandas as pd
import numpy as np
import seaborn as sb 
import matplotlib.pyplot as plt
from sklearn import datasets

data = datasets.load_iris()
data.DESCR

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.columns = df.columns.str.replace(' ', '_')

#EDA

df.info() # пропусков нет, масштаб один
sb.heatmap(df.corr(), cmap="YlGnBu", annot=True) # сильная линейная корреляция, можно предположить, что данные хороошо делятся

# Предобработка

from sklearn.preprocessing import StandardScaler

df.drop('target', axis= 1 , inplace= True )
scaler = StandardScaler()
model = scaler.fit(df)
scaled_data = model.transform(df)

df['sepal_length_(cm)'].plot(kind="hist")

df = pd.DataFrame(scaled_data, columns=data.feature_names)
df.columns = df.columns.str.replace(' ', '_')
df['sepal_length_(cm)'].plot(kind="hist")
df['target'] = data.target

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.33, random_state=42)
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
