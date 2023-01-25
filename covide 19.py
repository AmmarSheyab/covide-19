# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 00:01:05 2022

@author: ASUS
"""

import pandas as pd
import numpy as np
#data['date'] = pd.to_numeric(pd.to_datetime(data['date']))
df=pd.read_csv('covid-19-cases2.csv')
print(df.info())

df.isnull().sum()


X=df.iloc[:,2:14].values
y=df.iloc[:,0:2].values

df['date'] = pd.to_numeric(pd.to_datetime(df['date']))

print(df['date'].head())
print('*'*50)

from sklearn.impute import SimpleImputer
simpleImputer=SimpleImputer(missing_values=np.nan,strategy='mean')
simpleImputer.fit(X[:,2:14])
X[:,2:14]=simpleImputer.transform(X[:,2:14])

simpleImputer=SimpleImputer(missing_values=0,strategy='mean')
simpleImputer.fit(X[:,2:14])
X[:,2:14]=simpleImputer.transform(X[:,2:14])


#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
y[:,0]=labelEncoder.fit_transform(y[:,0])






def accuracy(a,b,n):
  import sklearn.metrics as sm
  print(f"\n------------ alograthim  {n} ----------------")

 # print(f"\n################# ACCURACY TEST NO.{c} ###################")
  
  print("Mean absolute error =", round(sm.mean_absolute_error(a,b), 2)) 
  print("Mean squared error =", round(sm.mean_squared_error(a,b), 2)) 
  print("Median absolute error =", round(sm.median_absolute_error(a, b), 2)) 
  print("Explain variance score =", round(sm.explained_variance_score(a, b), 2))



from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

X_forset_train,X_forest_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

regressor= MultiOutputRegressor(RandomForestRegressor(n_estimators=40,random_state=(42)))

regressor.fit(X_forset_train,y_train)

y_forest_pred=regressor.predict(X_forest_test)

    
accuracy(y_test,y_forest_pred,'RandomForestRegressor')



from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X_tree_train,X_tree_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
reg= MultiOutputRegressor(DecisionTreeRegressor(random_state=(42)))
reg.fit(X_tree_train,y_train)
y_tree_pred=reg.predict(X_tree_test)

accuracy(y_test,y_tree_pred,'DecisionTreeRegressor')

from sklearn.linear_model import Ridge

ridge=Ridge()

from sklearn.model_selection import train_test_split

X_ridge_train,X_ridge_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

ridge= MultiOutputRegressor(Ridge(random_state=(42)))

ridge.fit(X_ridge_train,y_train)

y_ridge_pred=regressor.predict(X_ridge_test)

    
accuracy(y_test,y_ridge_pred,'ridge')


from sklearn.svm import SVR
regressor= MultiOutputRegressor(SVR(kernel='rbf'))

from sklearn.model_selection import train_test_split

X_svr_train,X_svr_test,y_svr_train,y_svr_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_svr_train = sc_x.fit_transform(X_svr_train)

X_svr_test = sc_x.transform(X_svr_test)

regressor.fit(X_svr_train,y_svr_train)

y_svr_pred=regressor.predict(X_svr_test)


accuracy(y_svr_test,y_svr_pred,'svr /rbf')










