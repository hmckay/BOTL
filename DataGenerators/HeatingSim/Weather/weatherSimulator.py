import os
import math as m
import numpy as np
import reicosky
import rainfall
import pandas as pd
import random
from random import gauss
from datetime import datetime

YEARS = ["2014","2015","2016"]
MAX = "MaxTemperatureC"
MIN = "MinTemperatureC"
COLS = ["GMT","MaxTemperatureC","MinTemperatureC","Events"]
COMMENT = "No generative parameters"
OUTPUT_FP = "./weatherDataWithNoise.csv"
NOISE_PROB = 0.3

def readData(year):
    cols = COLS
    weather = pd.read_csv('WeatherData/weather'+str(year)+".csv",header=0, usecols=cols,index_col=0)
    sun = pd.read_csv('WeatherData/sun'+str(year)+".csv",header=None,skiprows=[0],names = range(1,13),index_col=0)
    avgRain = pd.read_csv('WeatherData/avgRainfall.csv',header=None)
    return weather,sun,avgRain

def addTempNoise(mean,var,noiseprob):
    if random.random() > (1-noiseprob):
        std = m.sqrt(var)
        noise = gauss(mean,std)
        return noise
    else: return mean

def addRainNoise(rain,noiseprob):
    if random.random()>(1-noiseprob):
        rain = (rain+1)%2
    return rain

def writeData(fp,date,time,temp,rain,comment):
    if not os.path.exists(fp):
        f = open(fp,'a')
        f.write("#DataRange: "+str(YEARS)+"\n")
        f.write("#"+comment+"\n")
        f.write("date,time,temp,rain\n")
    else:
        f = open(fp,'a')
    
    f.write(date+','+str(time)+','+str(temp)+','+str(rain)+'\n')
    f.close()


def main():
    
    for year in YEARS:
        weatherData,sunData,avgRain = readData(year)

        for index,d in weatherData.iterrows():
            date = datetime.strptime(index,"%Y-%m-%d")
            day = date.day
            month = date.month
            sunrise = sunData.loc[day,month]
            avgMonthRain,total,normalisedRain = rainfall.avgRainfall(avgRain,month)
            for time in range(100,2401,100):
                halfTemp = reicosky.reicosky(sunrise,d[MAX],d[MIN],time-70)
                hourTemp = reicosky.reicosky(sunrise,d[MAX],d[MIN],time)
                halfRain = rainfall.isRaining(d["Events"],avgMonthRain,total,normalisedRain)
                hourRain = rainfall.isRaining(d["Events"],avgMonthRain,total,normalisedRain)
                writeData(OUTPUT_FP,index,time-70,addTempNoise(halfTemp,random.random(),NOISE_PROB),addRainNoise(halfRain,NOISE_PROB),COMMENT)
                writeData(OUTPUT_FP,index,time,addTempNoise(hourTemp,random.random(),NOISE_PROB),addRainNoise(hourRain,NOISE_PROB),COMMENT)

if __name__ == "__main__":main()
            

