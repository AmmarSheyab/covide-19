# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:00:02 2022

@author: user
"""

import pandas as pd
import numpy as np
import sklearn.metrics as met
    
df = pd.read_csv('covid-19-cases2.csv',parse_dates=True)

df['date'] = pd.to_numeric(pd.to_datetime(df['date']))

def info():

    print("\n################### DATASET INFO ######################\n")

    df.info()  
    
    print("\n####################################################")
    print("\n####################################################")
    print("\n####################################################\n")

    i=0
    for column in df.columns:
        
        print(i,pd.api.types.infer_dtype(df[column]),
              "\t\tUniques:",df.iloc[:,i].nunique())
        
        i+=1
        
info()

X = df.iloc[:,2:].values

y = df.iloc[:,[0,1]].values

# Missing values: 13, 12, 11, 10, 9, 8

from sklearn.impute import SimpleImputer

simpleimputer = SimpleImputer(strategy='mean')

simpleimputer.fit(X)

X = simpleimputer.transform(X)

# If we want to make sure we no longer have 'Null' values:
    
print(pd.DataFrame(X).isnull().sum())

# No mixed data.



# DecisionTree:
    
from sklearn.multioutput import MultiOutputRegressor
  
from sklearn.model_selection import train_test_split    
  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeRegressor

regressor = MultiOutputRegressor(DecisionTreeRegressor(random_state=0))

regressor.fit(X_train,y_train)

y_dtr_pred = regressor.predict(X_test)

def accuracy(a,b,c):

    print(f"\n################# ACCURACY TEST NO.{c} ###################")

    print("\nMean absolute error:",round(met.mean_absolute_error(a, b),2))
    print("Mean absolute error%:",1-met.mean_absolute_error(a, b)/y_test.mean())

    print("\nVariance score:",round(met.explained_variance_score(a, b),5))

    print("\nMedian absolute error:",round(met.median_absolute_error(a, b),2))
    print("Median absolute error%:",1-met.median_absolute_error(a, b)/y_test.mean(),'\n')
    
print("############### DECISION TREE #################")    
accuracy(y_test,y_dtr_pred,1)

# RandomForest:
    
from sklearn.ensemble import RandomForestRegressor

regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=30,random_state=0))

regressor.fit(X_train,y_train)

y_rfr_pred = regressor.predict(X_test)

accuracy(y_test,y_rfr_pred,2)

# SVR rbf:

from sklearn.svm import SVR

regressor = MultiOutputRegressor(SVR(kernel='rbf'))

from sklearn.model_selection import train_test_split

X_svr_train,X_svr_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()

X_svr_train = sc_x.fit_transform(X_svr_train)

X_svr_test = sc_x.transform(X_svr_test)

y_train = sc_y.fit_transform(y_train)

y_test = sc_y.transform(y_test)

regressor.fit(X_svr_train,y_train)

y_rbf_pred = regressor.predict(X_svr_test)

accuracy(y_test,y_rbf_pred,3)

# SVR Linear:

regressor = MultiOutputRegressor(SVR(kernel='linear'))

regressor.fit(X_svr_train,y_train)

y_lin_pred = regressor.predict(X_svr_test)

accuracy(y_test,y_lin_pred,4)

# Ridge:
    
from sklearn.linear_model import Ridge

regressor = MultiOutputRegressor(Ridge(random_state=0))

regressor.fit(X_train,y_train)

y_rid_pred = regressor.predict(X_test)

accuracy(y_test,y_rid_pred,5)









































