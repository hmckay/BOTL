from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
import random

def avgRainfall(monthData,month):
    df = monthData#pd.read_csv("../../WeatherData/UK_Coventry/avgRainfall.csv",header=None)
    rain = df.iloc[month-1][0]
    total = df.sum()[0]
    normalisedRain = (rain-df.min()[0]*0.6)/(df.max()[0]-(df.min()[0]/2.5))
    return rain,total,normalisedRain

def isRaining(rainEvent, avgRainfall, totRainfall,normalised):
    raining = 0
    probRain = random.random()
    fracRain = avgRainfall/totRainfall
    #normalised +=0.1
    #print probRain
    #print fracRain
    
    if rainEvent:
        #print probRain
        #print probRain
        #print normalised
        if probRain < (normalised):
            raining = 1
    
    return raining

def dayData(date,rainDay):
    month = date.month
    avgMth,tot,norm = avgRainfall(month)
    
    for time in range(100,2401,100):
        rainHalf = isRaining(rainDay,avgMth,tot,norm)
        rainHour = isRaining(rainDay,avgMth,tot,norm)
        print str(time-70) + ": " + str(rainHalf)
        print str(time) + ": " + str(rainHour)
    
#dayData(datetime.strptime("2012-3-01","%Y-%m-%d"), 1)
#dayData("2012-01-01", 1)

