import pandas as pd
import numpy as np
import datetime
import calendar

def pullData(fp):
    allData = pd.read_csv(fp,header=0,date_parser='Time')
    allData = allData.dropna(axis=0)
    allData.reset_index(drop=True,inplace=True)
    allData.rename(index=str,columns={"ACC_CIP_TARGET_RA_2":"ACC"},inplace=True)
    allData = drop150ACC(allData)
    
    return allData

def drop150ACC(df):
    df = df[df["ACC"] != 150]
    df.reset_index(drop=True,inplace=True)
    return df

def subsetData(df,dFrom,dTo):
    #mask = (df['date']>=dFrom) & (df['date'] <= dTo)
    subsetDF = df.iloc[dFrom:dTo].copy()
    return subsetDF

def getWindowLength(df,dFrom,dTo):
    #mask = (df['date']>=dFrom) & (df['date'] < dTo)
    #subsetDF = df.loc[mask].copy()
    subsetDF = df.iloc[dFrom:dTo].copy()
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


