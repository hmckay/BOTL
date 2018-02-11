import pandas as pd
import numpy as np
import datetime
import calendar

def pullData(fp):
    allData = pd.read_csv(fp,header=0,comment='#')
    
    allData['date'] = pd.to_datetime(allData['date'],format="%Y-%m-%d")
    allData['year'] = allData['date'].dt.year
    allData['month']= allData['date'].dt.month
    allData['day'] = allData['date'].dt.day
    allData['dayOfWeek'] = allData['date'].dt.weekday
    return allData

def subsetData(df,dFrom,dTo):
    mask = (df['date']>=dFrom) & (df['date'] <= dTo)
    subsetDF = df.loc[mask].copy()
    return subsetDF

def getWindowLength(df,dFrom,dTo):
    mask = (df['date']>=dFrom) & (df['date'] < dTo)
    subsetDF = df.loc[mask].copy()
    return len(subsetDF)

def getWindow(df,windowSize):
    window = df.head(windowSize).copy()
    remainder = df[~df.isin(window)].dropna(axis = 0,how = 'all')
    #print remainder
    return window,remainder

def getNextEvalPeriod(periodInterval,startTime,startDate,df):
    subset = df.DataFrame() # may need to put column titles in here
    if startTime+periodInterval >2400:
        pass
    else:
        today = df[df['date']==startDate]
        mask = (df['time']>=startTime & df['time'<=startTime+600])
        subset = df.loc[mask].copy()
    return subset

def calcDate(now,d):
    return now + datetime.timedelta(days=d)


