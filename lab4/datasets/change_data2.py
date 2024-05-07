from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
df = pd.read_csv('titanic.csv')

onehotencoder = OneHotEncoder(sparse_output= False, drop='first')
encoded_df = pd.DataFrame(onehotencoder.fit_transform(df[['Embarked']]))

encoded_df.columns = onehotencoder.get_feature_names_out()
df_onehot = df.join(encoded_df)
df_onehot.drop('Embarked', axis = 1, inplace = True)

df_onehot.to_csv('titanic.csv', index=False)
