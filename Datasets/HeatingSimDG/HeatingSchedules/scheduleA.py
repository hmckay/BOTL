import pandas as pd
import math as m
import os
import calendar as c
import random
from random import gauss
from HeatingDistributions import midHeatDist as dist

#THRESHOLD probably 8, BASETEMP probably 12
#maybe pass prob of being home as a prarmeter

def weekday(time,oTemp,rain,THRESHOLD,BASETEMP):
    if 500 < time and time < 730:
        probHome = 0.98
        #if probHome > random.random():
        return dist.morning(oTemp,rain,0,THRESHOLD,THRESHOLD*2)
    elif 1800 < time and time < 2100:
        probHome = 0.9
        minOTemp = THRESHOLD + 2
        maxOTemp = THRESHOLD*2 + 1
        #if probHome > random.random():
        return dist.mid(oTemp,rain,0,minOTemp,maxOTemp)
    elif 2100 < time and time < 2330:
        probHome = 0.98
        #if probHome > random.random():
        return dist.night(oTemp,rain,0,THRESHOLD+2)
    return dist.noHeat(oTemp,THRESHOLD,BASETEMP)
    

def weekend(time,oTemp,rain,THRESHOLD,BASETEMP):
    if 630 < time and time < 900:
        probHome = 0.8
        #if probHome > random.random():
        return dist.night(oTemp,rain,4,THRESHOLD*1.5)
    elif 1300 < time and time < 1400:
        probHome = 0.4
        #if probHome > random.random():
        return dist.night(oTemp,False,2,THRESHOLD*1.5) 
    elif 1600 < time and time < 1700:
        probHome = 0.6
        #if probHome > random.random():
        return dist.night(oTemp,False,2,THRESHOLD*1.5)
    elif 1800 < time and time < 2100:
        probHome = 0.7
        minOTemp = THRESHOLD + 2
        maxOTemp = THRESHOLD*2 + 1
        #if probHome > random.random():
        return dist.mid(oTemp,rain,-2,minOTemp,maxOTemp)
    elif 2100 < time and time < 2400:
        probHome = 0.85
        #if probHome > random.random():
        return dist.night(oTemp,rain,-1,THRESHOLD+2)
    return dist.noHeat(oTemp,THRESHOLD,BASETEMP)

def holidayAway(oTemp,THRESHOLD,BASETEMP):
    return dist.noHeat(oTemp,THRESHOLD,BASETEMP)

def holidayHome(time,oTemp,THRESHOLD,BASETEMP):
    return weekend(time,oTemp,THRESHOLD,BASETEMP)

def addTempNoise(mean,var,NOISEPROB):
    if random.random() > (1-NOISEPROB) and mean > 0:
        std = m.sqrt(var)
        noise = gauss(mean,std)
        return noise
    else: return mean


#pass date as datetime object - form Y-m-d
def isWeekDay(date,time,oTemp,rain,THRESHOLD,BASETEMP,NOISEPROB,NOISEVAR):
    day = date.day
    month = date.month
    year = date.year
    if c.weekday(year,month,day)<5:
        temp,on = weekday(time,oTemp,rain,THRESHOLD,BASETEMP)
        if on:
            temp = addTempNoise(temp,NOISEVAR,NOISEPROB)
        return temp,on
    else:
        temp,on = weekend(time,oTemp,rain,THRESHOLD,BASETEMP)
        if on:
            temp = addTempNoise(temp,NOISEVAR,NOISEPROB)
        return temp,on
