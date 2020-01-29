#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:10:35 2019

@author: shubhamsharma
"""



import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_csv("./train.csv")

test_temp = pd.read_csv("./test.csv")

df.head()

#checking missing values
a=df.isna().sum()
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import voting_classifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import os

test = test_temp.drop(columns = 'ID')
 
df_x = df.drop(columns = 'class')
x_cols = df_x.columns
#df = df[(np.abs(stats.zscore(df_train)) < 3).all(axis=1)]
 
#df_train = df_train[(np.abs(stats.zscore(df_train)) < 3).all(axis=1)]
 
 
#df_train_normalizer = preprocessing.Normalizer().fit(df_train)
#df_train_normalized = df_train_normalizer.transform(df_train)
#df_train_normalized = pd.DataFrame(df_train_normalized)
#df_train_normalized.columns = df_train.columns
mmScaler = MinMaxScaler()
mmScaler.fit(df_x)
df_x =  pd.DataFrame(mmScaler.transform(df_x))
df_x.columns = x_cols
df_y = df['class']
 
#svc = LinearSVC(C = 0.01, penalty="l1", dual = False).fit(df_train_normalized,y_train)
#model = SelectFromModel(svc,prefit=True)
 
#df_train_final = model.transform(df_train_normalized)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.1, random_state=42, stratify = df_y)
 
model1 = RandomForestClassifier(n_estimators = 150, criterion="entropy", max_depth= 35, max_features= 10)
ne = 40
lr = 2
max_depth = 4
for i in range(40,100):
    print(i,lr,max_depth)
    model2 = GradientBoostingClassifier(n_estimators= i,learning_rate= lr, max_depth = max_depth)
    run_model(model2,  X_train, y_train, X_test, y_test)
    
   
model2 = GradientBoostingClassifier(n_estimators= 1550,learning_rate= 0.041, max_depth = 4)
run_model(model2,  X_train, y_train, X_test, y_test)
       model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn,fp,fn,tp = confusion_matrix(y_pred= y_pred, y_true= y_test).ravel()
    print("False Negative Rate", fn/(fn+tp)) 
   
model3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(64, 64), random_state=1)
 
run_model(model1,  X_train, y_train, X_test, y_test)
 
run_model(model3,  X_train, y_train, X_test, y_test)
model3.fit(X_train, y_train)
y_T = pd.DataFrame(model2.predict_proba(test))
 
#y_submit = pd.concat(y_T[0],axis = 1)
estimators = []
estimators.append(('GB', model2))
estimators.append(('NN', model3))
# create the ensemble model
model4 = voting_classifier.VotingClassifier(estimators, voting = "soft")
 
run_model(model4,  X_train, y_train, X_test, y_test)
 
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn,fp,fn,tp = confusion_matrix(y_pred= y_pred, y_true= y_test).ravel() 
    print("False Negative Rate", fn/(fn+tp))
    print("Accuracy",(tp+tn)/(tp+tn+fp+fn))
    
    
    

model2 = GradientBoostingClassifier(n_estimators= 67,learning_rate= 2, max_depth = 4)
run_model(model2,  X_train, y_train, X_test, y_test)    