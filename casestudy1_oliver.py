#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:30:55 2019

@author: oliver
"""
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pydot
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel





#import data into dataframe

df = pd.read_csv('CaseStudyData.csv')
rs = 10
ros = RandomOverSampler(random_state=0)
rus = RandomUnderSampler(random_state=0)
smote = SMOTE(random_state = 0)

#-----------------------------------------------------------------------------
#Task 1.1 KICK proportion
#-----------------------------------------------------------------------------
def Kick_proportion(df):
    ###count bad buy 0 = No 1 = Yes
    proportion = df['IsBadBuy'].value_counts()

    kick_proportion = proportion.iloc[1] / len(df)
    
    return kick_proportion
BADBUY = Kick_proportion(df)
#-----------------------------------------------------------------------------
#1.2 Fix Data quality problems
#-----------------------------------------------------------------------------

#replace ? to NAN
df = df.replace(['?','#VALUE!'], np.nan)    

#nominal colums replace with mode
nominal_cols = ['Auction','Make','Color','Transmission',
                'WheelTypeID','WheelType','Nationality','Size',
                'TopThreeAmericanName','PRIMEUNIT','AUCGUART',
                'VNST','IsOnlineSale','ForSale']


#numeric replace with median
num_cols = ['VehYear','VehOdo','MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
               'MMRCurrentRetailAveragePrice' , 'MMRCurrentRetailCleanPrice',
               'VehBCost','WarrantyCost']
#-----------------------------------------------------------------------------

def preprocess_data(df):
    
    ##Convert Manual to MANUAL
    df['Transmission'] = df['Transmission'].replace('Manual', 'MANUAL')

    # Replace USA to American
    df['Nationality'] = df['Nationality'].replace('USA', 'AMERICAN')
    
    ## ForSale
    # Drop 0 and convert to lower case
    df = df.drop(df[df.IsOnlineSale == '0'].index)
    df['ForSale']=df['ForSale'].str.lower()
   
    ## Convert 0.0 to 0 
    ###### -1 drop
    ## drop others
    df['IsOnlineSale'] = df['IsOnlineSale'].astype(str).replace('0.0','0')
    df['IsOnlineSale'] = df['IsOnlineSale'].replace(['1.0'], '1')
    df = df.drop(df[df.IsOnlineSale == '4.0'].index)
    df = df.drop(df[df.IsOnlineSale == '2.0'].index)
    df = df.drop(df[df.IsOnlineSale == '-1.0'].index)
    
    # MMRAcquisitionAuctionAveragePrice , 501 '0' PRICE
    # MMRAcquisitionAuctionCleanPrice, 414  '0' PRICE
    # MMRAcquisitionRetailAveragePrice, 501 '0' PRICE
    # MMRAcquisitonRetailCleanPrice ,  500 '0' PRICE

    # MMRCurrentAuctionAveragePrice  287 '0' Price
    # MMRCurrentAuctionCleanPrice   206
    # MMRCurrentRetailAveragePrice  287
    # MMRCurrentRetailCleanPrice    287
    # convert str to float to calucate RATIO
    
    price_col = ['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
                 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                 'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
                 'MMRCurrentRetailAveragePrice' , 'MMRCurrentRetailCleanPrice']
    # mask for drop prices 0 , 1    
    for i in price_col:
        df[i] = df[i].astype(float)
        mask = df[i] < 100
        df.loc[mask,i] = np.nan
    
    for i in num_cols:
       df[i] = df[i].astype(float)
     

    return df

#------------------------------------------------------------------------------
# using mode for nominal columns and median for numerical columns
# drop columns
def missing_values(df):
    for i in num_cols:
        df[i] = df[i].fillna(df[i].median())
        
    for i in nominal_cols:
        mode = df[i].mode()[0]
        df[i] = df[i].fillna(mode)
    
    #calucate ratio
    df['MMRCurrentRetailRatio'] = df['MMRCurrentRetailAveragePrice'] / df['MMRCurrentRetailCleanPrice']
    df['MMRCurrentRetailRatio'] = df['MMRCurrentRetailRatio'].round(4)
        
    #DROP DATA
    #BAD DATA
    df = df.drop(['PRIMEUNIT','AUCGUART'], axis = 1)

    #Same Data
    df = df.drop(['ForSale','IsOnlineSale',], axis = 1)
    
    
    #Irrelevant
    df = df.drop(['PurchaseID','PurchaseTimestamp','Color','WheelType'],axis = 1)

    # Convert To Date only

    df['PurchaseYear'] = pd.to_datetime(df['PurchaseDate']).dt.strftime('%Y')
    df['PurchaseMonth'] = pd.to_datetime(df['PurchaseDate']).dt.strftime('%m')
    
    col_time = ['PurchaseYear','PurchaseMonth']
    for i in col_time:
        df[i] = df[i].astype(int)
        
    df = df.drop('PurchaseDate', axis = 1)


    df = pd.get_dummies(df)
    
    return df

def split_data_for_dt(df):
    y = df['IsBadBuy']
    X = df.drop(['IsBadBuy'], axis = 1)
    X_mat = X.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y,test_size = 0.3, stratify = y, random_state = rs)
    
    #oversample 
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

def calculate_num_leaves(dt):
    n_nodes = dt.tree_.node_count
    ll = dt.tree_.children_left
    rl = dt.tree_.children_right
    count = 0
    for i in range(0,n_nodes):
        if (ll[i] & rl[i]) == -1:
            count = count + 1
    return count


# grab feature importances from the model and feature name from the original X
def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])
        
def split_data_for_RF_CNN(df_ready, num_cols):
    df_log = df_ready.copy()

    for col in num_cols:
        df_log[col] = df_log[col].apply(lambda x: x+1)
        df_log[col] = df_log[col].apply(np.log)

    y_log = df_log['IsBadBuy']
    X_log = df_log.drop(['IsBadBuy'], axis=1)
    X_mat_log = X_log.as_matrix()
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                    random_state=rs)
    X_train_log, y_train_log = ros.fit_resample(X_train_log, y_train_log)

    # standardise them again
    scaler_log = StandardScaler()
    X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
    X_test_log = scaler_log.transform(X_test_log)
    
    return X_train_log, X_test_log,y_train_log, y_test_log


processed_data = preprocess_data(df)

df_ready = missing_values(processed_data)

#------------------------------------------------------------------------------
# DT
#------------------------------------------------------------------------------
# change to the dummy


X_train, X_test, y_train, y_test = split_data_for_dt(df_ready)

#------------------------------------------------------------------------------
# Oversamping
#print('Original dataset shape %s' % Counter(y))
#print('Resampled dataset shape %s' % Counter(y_res))

#training
dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 9, min_samples_leaf = 2,random_state=rs)


dt.fit(X_train,y_train)
y_pred2 = dt.predict(X_test)
print("DT Default Train accuracy:", dt.score(X_train, y_train))
print("DT Default Test accuracy:", dt.score(X_test, y_test))
print(classification_report(y_test,y_pred2))

#size of nude
print(print("Number of nodes: ",dt.tree_.node_count))

# print leaves
print("The number of leaves is ",calculate_num_leaves(dt))
##------------------------------------------------------------------------------
# GIRDSEARCH
#------------------------------------------------------------------------------
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(7,10),
          'min_samples_leaf': range(2,3)}

cv_dt = GridSearchCV(param_grid= params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv_dt.fit(X_train, y_train)
print('strating')
print("DT GridSearch Train accuracy:", cv_dt.score(X_train, y_train))
print("DT GridSearch test accuracy:", cv_dt.score(X_test, y_test))

    # test the best model
y_pred = cv_dt.predict(X_test)
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv_dt.best_params_)

#------------------------------------------------------------------------------
# Feature Importance and visualising
#------------------------------------------------------------------------------
analyse_feature_importance(dt, X.columns,5)

# visualize
dotfile = StringIO()
export_graphviz(dt, out_file=dotfile, feature_names=X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph[0].write_png("dt_search.png") # saved in the following file - will return True if successful

#------------------------------------------------------------------------------
# Visualising relationship between hyperparameters and model performance
#------------------------------------------------------------------------------
test_score = []
train_score = []

# check the model performance for max depth from 2-20
for max_depth in range (2, 300):
    model = DecisionTreeClassifier(min_samples_leaf = max_depth, random_state=rs)
    model.fit(X_train, y_train)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))    

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 300), train_score, 'b', range(2,300), test_score, 'r')
plt.xlabel('min_samples_leaf\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')
plt.show()    
#import seaborn as sns    
#import matplotlib.pyplot as plt
#categoryCol = ['PurchaseDate','PurchaseTimestamp']    
    
#for i in categoryCol:
#    sns.countplot(data=df,x=i,hue="IsBadBuy")
#    plt.show()

#------------------------------------------------------------------------------
# LR
#------------------------------------------------------------------------------

X_train_log, X_test_log, y_train_log, y_test_log = split_data_for_RF_CNN(df_ready,num_cols)
model_rfe= LogisticRegression(C = 10, random_state=rs)

# fit it to training data
model_rfe.fit(X_train_log, y_train_log)
print("LR Default Train accuracy:", model_rfe.score(X_train_log, y_train_log))
print("LR Default Test accuracy:", model_rfe.score(X_test_log, y_test_log))

y_pred = model_rfe.predict(X_test_log)
print(classification_report(y_test_log, y_pred))
#------------------------------------------------------------------------------
#importantce
# grab feature importances from the model and feature name from the original X
coef = cv_lr.best_estimator_.coef_[0]
feature_names = X.columns

# sort them out in descending order
indices = np.argsort(np.absolute(coef))
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:100]

for i in indices:
    print(feature_names[i], ':', coef[i])

#------------------------------------------------------------------------------
# LR GridSearch
#------------------------------------------------------------------------------
params = {'C': [pow(10, x) for x in range(-8, 0)]}

# use all cores to tune logistic regression with C parameter
cv_lr = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv_lr.fit(X_train_log, y_train_log)

# test the best model
print("LR GridSearch Train accuracy:", cv_lr.score(X_train_log, y_train_log))
print("LR GridSearch Test accuracy:", cv_lr.score(X_test_log, y_test_log))

y_pred = cv_lr.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

# print parameters of the best model
print(cv_lr.best_params_)

##------------------------------------------------------------------------------
#   RFE feature selection
rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
rfe.fit(X_train_log, y_train_log)

print("Original feature set", X_train.shape[1])
print("Number of features after elimination", rfe.n_features_)

X_train_sel = rfe.transform(X_train_log)
X_test_sel = rfe.transform(X_test_log)

params = {'C': [pow(10, x) for x in range(-4, 4)]}

cv_rfe = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv_rfe.fit(X_train_sel, y_train_log)

# test the best model
print("LR_RFE Train accuracy:", cv_rfe.score(X_train_sel, y_train_log))
print("LR_RFE Test accuracy:", cv_rfe.score(X_test_sel, y_test_log))

y_pred = cv_rfe.predict(X_test_sel)
print(classification_report(y_test, y_pred))

# print parameters of the best model
print(cv_rfe.best_params_)
##------------------------------------------------------------------------------
#   DT feature selection
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(8,12),
          'min_samples_leaf': range(2,3)}

dt_sel = GridSearchCV(param_grid= params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
dt_sel.fit(X_train_log, y_train_log)

selectmodel = SelectFromModel(dt_sel.best_estimator_, prefit=True)
X_train_sel_model = selectmodel.transform(X_train_log)
X_test_sel_model = selectmodel.transform(X_test_log)

print(X_train_sel_model.shape)

params = {'C': [pow(10, x) for x in range(-6, 4)]}

cv_lr_dt = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv_lr_dt.fit(X_train_sel_model, y_train_log)

print("LR DT Train accuracy:", cv_lr_dt.score(X_train_sel_model, y_train_log))
print("LR DT Test accuracy:", cv_lr_dt.score(X_test_sel_model, y_test_log))

# test the best model
y_pred = cv_lr_dt.predict(X_test_sel_model)
print(classification_report(y_test_log, y_pred))

# print parameters of the best model
print(cv_lr_dt.best_params_)


#graph plot
test_score = []
train_score = []
params = [pow(10, x) for x in range(-10, 10)]
# check the model performance for max depth from 2-20
for c in params:
    model1 = LogisticRegression(C= c, random_state=rs)
    model1.fit(X_train_log, y_train_log)
    
    test_score.append(model1.score(X_test_log, y_test_log))
    train_score.append(model1.score(X_train_log, y_train_log))
for i in test_score:
    print(i)
import matplotlib.pyplot as plt

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(params, train_score,
        'b', params, test_score, 'r')
plt.xscale('log')
plt.xlabel('C\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')
plt.show()

#------------------------------------------------------------------------------
# MLP 
#------------------------------------------------------------------------------

X_train_log, X_test_log, y_train_log, y_test_log =split_data_for_RF_CNN(df_ready, num_cols)

mlp = MLPClassifier(random_state=rs)
mlp.fit(X_train_log, y_train_log)

print("Train accuracy:", mlp.score(X_train_log, y_train_log))
print("Test accuracy:", mlp.score(X_test_log, y_test_log))

y_pred = mlp.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

print(model)

#------------------------------------------------------------------------------
# MLP GridSearch
#------------------------------------------------------------------------------
X_train_log.shape
# X_train features = neurons
params = {'hidden_layer_sizes': (105,) , 'alpha': [0.01,0.001, 0.0001, 0.00001]}

cv_mlp = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv_mlp.fit(X_train_log, y_train_log)

print("Train accuracy:", cv_mlp.score(X_train_log, y_train_log))
print("Test accuracy:", cv_mlp.score(X_test_log, y_test_log))

y_pred = cv_mlp.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

print(cv_mlp.best_params_)

##------------------------------------------------------------------------------
# Feature Selection DT CNN

#--------------------

X_train_log, X_test_log, y_train_log, y_test_log =split_data_for_RF_CNN(df_ready,num_cols)

params = {'criterion': ['gini', 'entropy'],
          'max_depth':range(11,12,2),
          'min_samples_leaf': range(2,3)}

dt_sel = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
dt_sel.fit(X_train_log, y_train_log)



selectmodel = SelectFromModel(dt_sel.best_estimator_, prefit=True)
X_train_sel_model = selectmodel.transform(X_train_log)
X_test_sel_model = selectmodel.transform(X_test_log)

print(X_train_sel_model.shape)




params = {'hidden_layer_sizes': (105,) , 'alpha': [0.01,0.001]}

cv_dt_cnn = GridSearchCV(param_grid=params, estimator=MLPClassifier( random_state=rs), cv=10, n_jobs=-1)
cv_dt_cnn.fit(X_train_sel_model, y_train_log)

print("Train accuracy:", cv_dt_cnn.score(X_train_sel_model, y_train))
print("Test accuracy:", cv_dt_cnn.score(X_test_sel_model, y_test))

y_pred = cv_dt_cnn.predict(X_test_sel_model)
print(classification_report(y_test, y_pred))

print(cv_dt_cnn.best_params_)

##------------------------------------------------------------------------------
# Feature Selection RF
#--------------------
X_train_sel = rfe.transform(X_train_log)
X_test_sel = rfe.transform(X_test_log)

params = {'hidden_layer_sizes': [(90,), (92,), (100,), (105,)], 'alpha': [0.0001, 0.00001]}

cv_cnn_rf= GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv_cnn_rf.fit(X_train_sel, y_train_log)

print("Train accuracy:", cv_cnn_rf.score(X_train_sel, y_train_log))
print("Test accuracy:", cv_cnn_rf.score(X_test_sel, y_test_log))

y_pred = cv_cnn_rf.predict(X_test_sel)
print(classification_report(y_test_log, y_pred))

print(cv_cnn_rf.best_params_)


##------------------------------------------------------------------------------
# ensemble model
#--------------------

dt_model = cv_dt.best_estimator_
print(dt_model)

cnn_model = cv_dt_cnn.best_estimator_
print(cnn_model)

cl_model = cv_rfe.best_estimator_
print(cl_model)



from sklearn.metrics import roc_auc_score

y_pred_proba_dt = dt_model.predict_proba(X_test)
y_pred_proba_log_reg = cl_model.predict_proba(X_test_sel)
y_pred_proba_nn = cnn_model.predict_proba(X_test_sel_model)

roc_index_dt = roc_auc_score(y_test, y_pred_proba_dt[:, 1])
roc_index_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg[:, 1])
roc_index_nn = roc_auc_score(y_test, y_pred_proba_nn[:, 1])



print("ROC index on test for DT:", roc_index_dt)
print("ROC index on test for logistic regression:", roc_index_log_reg)
print("ROC index on test for NN:", roc_index_nn)



from sklearn.metrics import roc_curve

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_proba_dt[:,1])
fpr_log_reg, tpr_log_reg, thresholds_log_reg = roc_curve(y_test, y_pred_proba_log_reg[:,1])
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_pred_proba_nn[:,1])
fpr_en, tpr_en,thresholds_en = roc_curve(y_test,y_pred_proba_ensemble[:,1])

import matplotlib.pyplot as plt

plt.plot(fpr_dt, tpr_dt, label='ROC Curve for DT {:.3f}'.format(roc_index_dt), color='red', lw=0.5)
plt.plot(fpr_log_reg, tpr_log_reg, label='ROC Curve for Log reg {:.3f}'.format(roc_index_log_reg), color='green', lw=0.5)
plt.plot(fpr_nn, tpr_nn, label='ROC Curve for NN {:.3f}'.format(roc_index_nn), color='darkorange', lw=0.5)
plt.plot(fpr_en, tpr_en, label = 'ROC Curve for ensemble {:.3f}'.format(roc_index_ensemble),color = 'blue',lw = 0.5)
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

##------------------------------------------------------------------------------
# Ensemble Modeling
#--------------------
# import the model
from sklearn.ensemble import VotingClassifier

# initialise the classifier with 3 different estimators
voting = VotingClassifier(estimators=[('dt', dt_model), ('lr', cl_model), ('nn', cnn_model)], voting='soft')

voting.fit(X_train_log, y_train_log)
y_pred_ensemble = voting.predict(X_test_log)
# evaluate train and test accuracy
print("Ensemble train accuracy:", voting.score(X_train_log, y_train_log))
print("Ensemble test accuracy:", voting.score(X_test_log, y_test_log))

# evaluate ROC auc score
y_pred_proba_ensemble = voting.predict_proba(X_test_log)
roc_index_ensemble = roc_auc_score(y_test_log, y_pred_proba_ensemble[:, 1])
print("ROC score of voting classifier:", roc_index_ensemble)
    

print(classification_report(y_test, y_pred_proba_ensemble))



print("\nReport for Ensemble: \n",classification_report(y_test_log, y_pred_ensemble))

