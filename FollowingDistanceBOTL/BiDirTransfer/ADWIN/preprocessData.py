import pandas as pd
import numpy as np
import datetime
import calendar

def pullData(fp):
    allData = pd.read_csv(fp,header=0,date_parser='Time')
    allData = allData.dropna(axis=0)
    allData.reset_index(drop=True,inplace=True)
    return allData

def subsetData(df,dFrom,dTo):
    subsetDF = df.iloc[dFrom:dTo].copy()
    return subsetDF

def getWindowLength(df,dFrom,dTo):
    subsetDF = df.iloc[dFrom:dTo].copy()
    return len(subsetDF)

def getWindow(df,windowSize):
    window = df.head(windowSize).copy()
    remainder = df[~df.isin(window)].dropna(axis = 0,how = 'all')
    return window,remainder

