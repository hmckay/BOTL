import pandas as pd
import numpy as np
import datetime
import calendar

def pullData(fp):
    allData = pd.read_csv(fp,header=0,comment='#')
    
    #allData['date'] = pd.to_datetime(allData['date'],format="%Y-%m-%d")
    #allData['year'] = allData['date'].dt.year
    #allData['month']= allData['date'].dt.month
    #allData['day'] = allData['date'].dt.day
    #allData['dayOfWeek'] = allData['date'].dt.weekday
    return allData

def subsetData(df,dFrom,dTo):
    #mask = (df['date']>=dFrom) & (df['date'] <= dTo)
    subsetDF = df.iloc[dFrom:dTo].copy()
    return subsetDF

def pullFollowingData(fp):
    allData = pd.read_csv(fp,header=0,date_parser='Time')
    allData = allData.dropna(axis=0)
    allData.reset_index(drop=True,inplace=True)
    return allData

def subsetFollowingData(df,dFrom,dTo):
    subsetDF = df.iloc[dFrom:dTo].copy()
    return subsetDF

def pullHeatingData(fp):
    allData = pd.read_csv(fp,header=0,comment='#')
    
    allData['date'] = pd.to_datetime(allData['date'],format="%Y-%m-%d")
    allData['year'] = allData['date'].dt.year
    allData['month']= allData['date'].dt.month
    allData['day'] = allData['date'].dt.day
    allData['dayOfWeek'] = allData['date'].dt.weekday
    allData = allData.astype({'time':'int64','rain':'int64','dayOfWeek':'int64'})
    print(allData.dtypes)
    return allData

def subsetHeatingData(df,dFrom,dTo):
    mask = (df['date']>=pd.Timestamp(dFrom)) & (df['date'] <= pd.Timestamp(dTo))
    subsetDF = df.loc[mask].copy()
    return subsetDF

def getWindowLength(df,dFrom,dTo):
    subsetDF = df.iloc[dFrom:dTo].copy()
    return len(subsetDF)

def getWindow(df,windowSize):
    window = df.head(windowSize).copy()
    remainder = df[~df.isin(window)].dropna(axis = 0,how = 'all')
    return window,remainder

