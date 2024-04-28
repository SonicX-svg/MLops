import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

temperature_df = pd.concat([train, test])

temperature_df = temperature_df[[c for c
        in list(temperature_df)
        if len(temperature_df[c].unique()) > 1]] #Перезаписываем датасет, оставляя только те колонки, в которых больше одного уникального значения На всякий

for i in temperature_df.columns:
  if temperature_df[i].isnull().sum() / temperature_df.shape[0] * 100 > 50:
    temperature_df.drop(labels=[i], axis=1, inplace=True)#Удаляем, если какая-то колонка имеет больше 50 пустых значений
  else:
    temperature_df.dropna(inplace=True) #Удаляем строчки с пустыми значениями, если потом останется достаточно данных для обучения
#print('12 ', temperature_df)   
temperature_df = temperature_df.loc[((temperature_df['Хороший_день'].isin([0,1])) & (temperature_df['Пол'].isin(['м','ж'])))] # Хороший_день и пол должны иметь 2 значения. убрали неврзможные значения    
#print('2 ', temperature_df)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(sparse_output = False, drop='first')
#print(temperature_df)
encoded_df = pd.DataFrame(onehotencoder.fit_transform(temperature_df[['Осадки']]))
encoded_df.rename(columns = {0:'ураган', 1:'дождь', 2:'снег',3:'без осадков'}, inplace = True )

temperature_df.reset_index(inplace=True, drop=True) # теперь, когда индексация при обьединении двух фреймов одинакова значаний nan не возникает. При удалении строк обязательно делать переиндексацию

df_onehot = temperature_df.join(encoded_df)
df_onehot.drop('Осадки', axis = 1, inplace = True)

encoded_df1 = pd.DataFrame(onehotencoder.fit_transform(temperature_df[['Пол']]))

df_onehot['Пол'] = encoded_df1

df_onehot = df_onehot.loc[((df_onehot['Возраст']>0) & (df_onehot['Возраст']<120))] # убрали аномалии в возрасте
# Убираем выбросы в температуре
df_onehot = df_onehot.loc[((df_onehot['Температура']>-50) & (df_onehot['Температура']<50))]

# делим данные на колич и качественные
df_numeric = df_onehot[['Температура', 'Возраст']]
df_nonnumeric = df_onehot.loc[:, ~df_onehot.columns.isin(['Температура', 'Возраст'])]
#print('aaaaaaaaaaaaaaaaaaaaaaaaa', df_onehot)
#Нормализуем числовые данные
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

scaler_temperatura = scaler.fit_transform(df_numeric[['Температура']])
scaler_age = scaler.fit_transform(df_numeric[['Возраст']])

df_onehot.update(scaler_temperatura) # обьеденяем обработанные чис и кач данные в единый сет
#print(df_onehot)
df_onehot.update(scaler_age)
from sklearn.model_selection import train_test_split
train, test =  train_test_split(df_onehot,test_size=0.33, random_state=42)
#print(train.shape)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
