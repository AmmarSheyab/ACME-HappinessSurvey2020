# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:40:57 2023

@author: ASUS
"""
import numpy as np
import pandas as pd

df=pd.read_csv('C:/Users/ASUS/Desktop/New folder (2)/ACME-HappinessSurvey2020.csv')
print('#################')
df.info()
print('********* chek  non value in dataset ********* ')
print('#################')
print(df.isnull().sum())
print('********* show 5 end row and colums in data *********')
print('#################')
print(df.tail())
print('********* show 5 row and columns in data *********')
print('#################')
print(df.head(5))
print('#################')

X = df.iloc[:,1:-1].values
y = df.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=(0))

from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train,y_train)
y_pred=linearRegression.predict(X_test)

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 
print("\nVariance score:",round(sm.explained_variance_score(y_test, y_pred),5))




