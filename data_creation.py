import pandas as pd
import numpy as np
import random
from copy import deepcopy

temperature = np.random.normal(loc=5, scale=7, size=1096)
temperature_df = pd.DataFrame({ 'Температура': temperature})
dfupdate=temperature_df.sample(100)
dfupdate['Температура']=np.random.normal(loc=5, scale=15, size=len(dfupdate))
temperature_df.update(dfupdate)
temperature_df['Осадки']=[random.randint(0, 3) for i in range(1096)]
temperature_df['Пол']=['м' if random.randint(0, 1) else 'ж' for i in range(1096)] # преобразование в категориальную аномалий
temperature_df['Возраст']=[random.randint(5, 70) for i in range(1080)]+[random.randint(-100, -1) for i in range(8)]+[random.randint(100, 1000) for i in range(8)] # добавление аномалий в возраст
temperature_df['Хороший_день']=[random.randint(0, 1) for i in range(1090)]+list(np.random.randint(0, 3, 6)) # добавление аномалий

import math

for i in range(3):
  sample = temperature_df.sample(20)
  sample.iloc[:, random.randint(0, 4)] = np.NaN
  df_concatenated = pd.concat([temperature_df, sample]) # используем concat, т.к update игнорирует Nan
  temperature_df=deepcopy(df_concatenated.loc[~df_concatenated.index.duplicated(keep='last')])
  temperature_df.sort_index(inplace=True)
#temperature_df.isnull().values.any()

from sklearn.model_selection import train_test_split

train, test = train_test_split(temperature_df, test_size= 0.2 , random_state= 0 )

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
