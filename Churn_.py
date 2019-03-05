# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:37:11 2019

@author: manju
"""

import pandas as pd

churn=pd.read_csv("Churn_Modelling.csv")

churn.head()

churn=churn.drop(['RowNumber','CustomerId','Surname'],axis=1)

Y=churn['Exited']

churn=churn.drop(['Exited'],axis=1)

import keras 
from keras.models import Sequential

from keras.layers import Dense

my_ann=Sequential()

my_ann.add(Dense(units=32,kernel_initializer='uniform',activation='relu',input_dim=10))

churn=churn.get_dummies(churn)

X=churn.drop(['Geography_France','Gender_Female'],axis=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


my_ann.add(Dense(units=32,kernel_initializer='uniform',activation='relu',input_dim=11))

----------------
##add hidden layer

my_ann.add(Dense(units=32,kernel_initializer='uniform',activation='relu'))

##output layer
my_ann.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

print(my_ann.summary())

my_ann.complile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from sklearn.model_selection import train_test_split
[xtrain,xtest,ytrain,ytest]=train_test_split(X,Y,test_size=0.3,random_state=42)

my_ann.fit(xtrain,ytrain,batch_size=10,epochs=100)

ypred=my_ann.predict(xtest)

ypred=(ypred>0.5)

from sklearn.metrics import accuracy_score

acc=accuracy_score(ytest,ypred)
print(acc)

