#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:30:55 2019

@author: oliver
"""
import pandas as pd
import numpy as np


#import data into dataframe

df = pd.read_csv('CaseStudyData.csv')
#-----------------------------------------------------------------------------
#Task 1.1 KICK proportion
#-----------------------------------------------------------------------------
def Kick_proportion():
    ###count bad buy 0 = No 1 = Yes
    proportion = df['IsBadBuy'].value_counts()

    kick_proportion = proportion.iloc[1] / len(df)
    
    return kick_proportion

#-----------------------------------------------------------------------------
#1.2 Fix Data quality problems
#-----------------------------------------------------------------------------

#replace ? to NAN
df = df.replace('?', np.nan)    

#nominal colums replace with mode
nominal_cols = ['Auction','VehYear','Make','Color','Transmission',
                'WheelTypeID','WheelType','Nationality','Size',
                'TopThreeAmericanName','PRIMEUNIT','AUCGUART',
                'VNST','IsOnlineSale','ForSale']


#numeric replace with median
num_cols = ['VehOdo','MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
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
    df = df.drop('ForSale', axis = 1)
    
#    df = df.drop('IsOnlineSale', axis = 1)
    
    #Irrelevant
    df = df.drop(['PurchaseID','PurchaseTimestamp'], axis = 1)
    
    
    return df


BADBUY = Kick_proportion()


processed_data = preprocess_data(df)

df_ready = missing_values(processed_data)





