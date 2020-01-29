#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:54:46 2019

@author: shubhamsharma
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
#from sklearn.ensemble import voting_classifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix, auc
from sklearn.neural_network import MLPClassifier
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import lightgbm as lgbm
import bayes_opt
import os
from imblearn.over_sampling import SMOTE
import seaborn as sn

ax2=sn.distplot(df[['Attr2']],hist=False)
df[['class']].hist()
ax3=sn.distplot(df[['class']],hist=False)
ax4=sn.distplot(df[['Attr4']],hist=False)
ax5=sn.distplot(df[['Attr5']],hist=False)
ax6=sn.distplot(df[['Attr6']],hist=False)
ax7=sn.distplot(df[['Attr7']],hist=False)
ax8=sn.distplot(df[['Attr8']],hist=False)
ax9=sn.distplot(df[['Attr2']],hist=False)
ax10=sn.distplot(df[['Attr2']],hist=False)
ax11=sn.distplot(df[['Attr2']],hist=False)
ax12=sn.distplot(df[['Attr2']],hist=False)
ax13=sn.distplot(df[['Attr2']],hist=False)
ax14=sn.distplot(df[['Attr2']],hist=False)
ax15=sn.distplot(df[['Attr2']],hist=False)
ax16=sn.distplot(df[['Attr2']],hist=False)
ax17=sn.distplot(df[['Attr2']],hist=False)
ax18=sn.distplot(df[['Attr2']],hist=False)
ax19=sn.distplot(df[['Attr2']],hist=False)
ax20=sn.distplot(df[['Attr2']],hist=False)
ax21=sn.distplot(df[['Attr2']],hist=False)
ax22=sn.distplot(df[['Attr2']],hist=False)
ax23=sn.distplot(df[['Attr2']],hist=False)
ax24=sn.distplot(df[['Attr2']],hist=False)
ax25=sn.distplot(df[['Attr2']],hist=False)
ax26=sn.distplot(df[['Attr2']],hist=False)


def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn,fp,fn,tp = confusion_matrix(y_pred= y_pred, y_true= y_test).ravel()
    print(confusion_matrix(y_pred= y_pred, y_true= y_test))
    print("Accuracy",(tn+tp)/(tn+tp+fp+fn))
    print("FPR", fp/(fp + tn))
    return fn/(fn+tp)
  
df = pd.read_csv('train.csv')
test_temp = pd.read_csv('test.csv')
test = test_temp.drop(columns = 'ID')
 
df_x = df.drop(columns = 'class')
x_cols = df_x.columns
#df = df[(np.abs(stats.zscore(df_x)) < 3).all(axis=1)]
df_x = df.drop(columns = 'class')
 
 
 
#df_train_normalizer = preprocessing.Normalizer().fit(df_train)
#df_train_normalized = df_train_normalizer.transform(df_train)
#df_train_normalized = pd.DataFrame(df_train_normalized)
#df_train_normalized.columns = df_train.columns
df_y = df['class']
sm = SMOTE(random_state = 42)
X_res, y_res = sm.fit_resample(df_x, df_y)
X_res = pd.DataFrame(X_res)
y_res = pd.DataFrame(y_res)
X_res.columns = x_cols
y_res.columns = ['class']
df_x_y = df_x.copy()
df_x_y['y'] = df_y
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.01, random_state=42, stratify = y_res)
 
 
model1 = RandomForestClassifier(n_estimators = 150, criterion="entropy", max_depth= 35, max_features= 10)
ne= [x for x in range(40,71)]
#lr = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
#model_scores = pd.DataFrame(columns=['ne', 'lr', 'md', 'fnr'])
model_log = LogisticRegression(C = 2)
model_GBM = GradientBoostingClassifier(n_estimators= 62 ,learning_rate= 1.5, max_depth = 3)
skf = StratifiedKFold(n_splits = 10, shuffle = True)
y_pred_cv = pd.DataFrame(cross_val_predict(model_GBM,X_test, y_test, cv = skf))
confusion_matrix(y_test, y_pred_cv)
 
#print(run_model(model_GBM,  X_train, y_train, X_test, y_test))
#for i in range(40,71):
#
##            max_depth = np.random.randint(low = 3, high = 5)
#    print(i,2,4)
#    model2 = GradientBoostingClassifier(n_estimators= i,learning_rate= 2, max_depth = 4)
#    fnr = run_model(model2,  X_train, y_train, X_test, y_test)
#    print(fnr)
#        
        
 
 
#model_final = GradientBoostingClassifier(n_estimators= 1500,learning_rate= 0.04, max_depth = 3)
#model_final = GradientBoostingClassifier(n_estimators= 1500,learning_rate= 0.04, max_depth = 4)
model_final = XGBClassifier(learning_rate = 0.01, max_depth = 4, n_estimators = 1500, scale_pos_weight = 46   )
model_final.fit(X_train, y_train)
print(run_model(model_final,  X_train, y_train, X_test, y_test))
 
l = lgbm.LGBMClassifier(scale_pos_weight = 1, n_estimators= 10000, max_depth=5, learning_rate=0.004, num_leaves= 45,
                        min_child_weight = 5,min_split_gain=0.038,reg_lambda = 2, subsample= 0.845 )
print(run_model(l,  X_train, y_train, X_test, y_test))
 
 
 
y_pred = pd.DataFrame(l.predict_proba(test))
#y_pred = pd.DataFrame(model_final.predict(df_test))
y_pred.columns = ['a', 'b']
 
y_pred[y_pred['a'] < 0.5]
 
y_pred.to_csv('t6.csv')
print(confusion_matrix(y_pred= y_pred, y_true= y_test))
#y_submit = pd.concat(y_T[0],axis = 1)
estimators = []
estimators.append(('XGBM', model_final))
estimators.append(('LGBM', l))
# create the ensemble model
model4 = voting_classifier.VotingClassifier(estimators, weights= [1,1])
y_pred = pd.DataFrame(model4.predict(test))
run_model(model4,  X_train, y_train, X_test, y_test)
 
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
def bayes_parameter_opt_lgb(X, y, init_round=10, opt_round=10, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.04, output_process=False):
    # prepare data
    train_data = lgbm.Dataset(data=X, label=y, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgbm.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range
    lgbBO = bayes_opt.BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
   
#    # output optimization process
#    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
   
    # return best parameters
    return lgbBO.res
 
opt_results = bayes_parameter_opt_lgb(df_x,df_y)