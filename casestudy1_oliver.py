#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:30:55 2019

@author: oliver
"""
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
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





#import data into dataframe

df = pd.read_csv('CaseStudyData.csv')
rs = 10
ros = RandomOverSampler(random_state=0)
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
    
#    for i in num_cols:
#        df[i] = df[i].astype(float)

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
    df = df.drop(['PurchaseID','PurchaseTimestamp','WheelTypeID','Color','WheelType'],axis = 1)
    # Convert To Date only

    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate']).dt.strftime('%m/%Y')

    df = pd.get_dummies(df)
    
    return df




processed_data = preprocess_data(df)

df_ready = missing_values(processed_data)

#------------------------------------------------------------------------------
# DT
#------------------------------------------------------------------------------
# change to the dummy

y = df_ready['IsBadBuy']
X = df_ready.drop(['IsBadBuy'], axis = 1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y,test_size = 0.2, stratify = y,
                                                    random_state = rs)
model = DecisionTreeClassifier(random_state=rs)

#training
model.fit(X_train, y_train)
print("DT Default Train accuracy:", model.score(X_train, y_train))
print("DT Default Test accuracy:", model.score(X_test, y_test))
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
#------------------------------------------------------------------------------
# Oversamping

X_res, y_res = ros.fit_resample(X_train, y_train)

print('Original dataset shape %s' % Counter(y))
print('Resampled dataset shape %s' % Counter(y_res))

#training
model.fit(X_res,y_res)
y_pred2 = model.predict(X_test)
print("DT Default resampled Train accuracy:", model.score(X_res, y_res))
print("DT Default resampled Test accuracy:", model.score(X_test, y_test))
print(classification_report(y_test,y_pred2))

##------------------------------------------------------------------------------
# GIRDSEARCH
#------------------------------------------------------------------------------
def DT_GirdSearch():
    params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 10),
          'min_samples_leaf': range(45,56)}

    cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
    cv.fit(X_train, y_train)
    print('strating')
    print("Train accuracy:", cv.score(X_train, y_train))
    print("Test accuracy:", cv.score(X_test, y_test))

    # test the best model
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred))

# print parameters of the best model
    print(cv.best_params_)

DT_GirdSearch()
#------------------------------------------------------------------------------
# Feature Importance
#------------------------------------------------------------------------------

# grab feature importances from the model and feature name from the original X
importances = model.feature_importances_
feature_names = X.columns

# sort them out in descending order
indices = np.argsort(importances)
indices = np.flip(indices, axis=0)

# limit to 20 features, you can leave this out to print out everything
indices = indices[:20]

for i in indices:
    print(feature_names[i], ':', importances[i])
    
    
    
    
    #
#------------------------------------------------------------------------------
# Visualising relationship between hyperparameters and model performance
#------------------------------------------------------------------------------

test_score = []
train_score = []

# check the model performance for max depth from 2-20
for max_depth in range(2, 20):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=rs)
    model.fit(X_res, y_res)
    
    test_score.append(model.score(X_test, y_test))
    train_score.append(model.score(X_train, y_train))    

# plot max depth hyperparameter values vs training and test accuracy score
plt.plot(range(2, 20), train_score, 'b', range(2,20), test_score, 'r')
plt.xlabel('max_depth\nBlue = training acc. Red = test acc.')
plt.ylabel('accuracy')
plt.show()    
#import seaborn as sns    
#import matplotlib.pyplot as plt
#categoryCol = ['PurchaseDate','PurchaseTimestamp']    
    
#for i in categoryCol:
#    sns.countplot(data=df,x=i,hue="IsBadBuy")
#    plt.show()


# visualize
dotfile = StringIO()
export_graphviz(model, out_file=dotfile, feature_names=X.columns)
graph = pydot.graph_from_dot_data(dotfile.getvalue())
graph[0].write_png("week3_dt_viz.png") # saved in the following file - will return True if successful

#------------------------------------------------------------------------------
# LR
#------------------------------------------------------------------------------

scaler = StandardScaler()
# learn the mean and std.dev of variables from training data
# then use the learned values to transform training data
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state=rs)

# fit it to training data
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

# classification report on test data
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
#------------------------------------------------------------------------------
# LR GridSearch
#------------------------------------------------------------------------------

df_log = df_ready.copy()



num_cols = ['VehYear','VehOdo','MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice',
               'MMRCurrentRetailAveragePrice' , 'MMRCurrentRetailCleanPrice',
               'VehBCost','WarrantyCost']
for col in num_cols:
    df_log[col] = df_log[col].apply(lambda x: x+1)
    df_log[col] = df_log[col].apply(np.log)

y_log = df_log['IsBadBuy']
X_log = df_log.drop(['IsBadBuy'], axis=1)
X_mat_log = X_log.as_matrix()
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_mat_log, y_log, test_size=0.3, stratify=y_log, 
                                                                    random_state=rs)

# standardise them again
scaler_log = StandardScaler()
X_train_log = scaler_log.fit_transform(X_train_log, y_train_log)
X_test_log = scaler_log.transform(X_test_log)

from sklearn.feature_selection import RFECV

rfe = RFECV(estimator = LogisticRegression(random_state=rs), cv=10)
rfe.fit(X_train_log, y_train_log) # run the RFECV

# comparing how many variables before and after
print("Original feature set", X_train.shape[1])
print("Number of features after elimination", rfe.n_features_)

params = {'C': [pow(10, x) for x in range(-6, 4)]}

cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_log, y_train_log)

# test the best model
print("Train accuracy:", cv.score(X_train_log, y_train_log))
print("Test accuracy:", cv.score(X_test_log, y_test_log))

y_pred = cv.predict(X_test_log)
print(classification_report(y_test_log, y_pred))

# print parameters of the best model
print(cv.best_params_)

params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(2, 7),
          'min_samples_leaf': range(20, 60, 10)}

cv = GridSearchCV(param_grid=params, estimator=DecisionTreeClassifier(random_state=rs), cv=10)
cv.fit(X_train_log, y_train_log)

print(cv.best_params_)


# use the trained best decision tree from GridSearchCV to select features
# supply the prefit=True parameter to stop SelectFromModel to re-train the model
selectmodel = SelectFromModel(cv.best_estimator_, prefit=True)
X_train_sel_model = selectmodel.transform(X_train_log)
X_test_sel_model = selectmodel.transform(X_test_log)

print(X_train_sel_model.shape)

params = {'C': [pow(10, x) for x in range(-6, 4)]}

cv = GridSearchCV(param_grid=params, estimator=LogisticRegression(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train_sel_model, y_train_log)

print("Train accuracy:", cv.score(X_train_sel_model, y_train_log))
print("Test accuracy:", cv.score(X_test_sel_model, y_test_log))

# test the best model
y_pred = cv.predict(X_test_sel_model)
print(classification_report(y_test_log, y_pred))

# print parameters of the best model
print(cv.best_params_)

#------------------------------------------------------------------------------
# MLP 
#------------------------------------------------------------------------------


y = df_ready['IsBadBuy']
X = df_ready.drop(['IsBadBuy'], axis=1)
X_mat = X.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.2, stratify=y, random_state=rs)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(max_iter=100, random_state=rs)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print(model)

#------------------------------------------------------------------------------
# MLP GridSearch
#------------------------------------------------------------------------------

# X_train features = neurons
print(X_train.shape)



# Find hidden_layer_sizes
params = {'hidden_layer_sizes': [(x,) for x in range(5, 100, 20)]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_params_)

params = {'hidden_layer_sizes': [(3,), (50,), (80,), (90,)], 'alpha': [0.01,0.001, 0.0001, 0.00001]}

cv = GridSearchCV(param_grid=params, estimator=MLPClassifier(random_state=rs), cv=10, n_jobs=-1)
cv.fit(X_train, y_train)

print("Train accuracy:", cv.score(X_train, y_train))
print("Test accuracy:", cv.score(X_test, y_test))

y_pred = cv.predict(X_test)
print(classification_report(y_test, y_pred))

print(cv.best_params_)





    