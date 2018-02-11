import pandas as pd 
import os 
import random
import numpy as np
import scheduleA
from datetime import datetime

WD_FP = '../Weather/weatherDataWithNoise.csv'
OUTPUT_FP = '../Datasets/userSOURCEDataSimulation.csv'
YEARS = ["2014","2015","2016"]
MIN_OUTSIDE_TEMP = 8
MIN_HEATING_TEMP = 12
NOISE_PROB = 0.5
COMMENT = "No generative parameters: min outside: 8, min heating temp: 12, noise_prob=0.5 - to be used as SOURCE"

def readData(fp):
    df = pd.read_csv(fp,header=0,comment='#')
    return df

def writeData(fp,date,time,oTemp,rain,dTemp,on,comment):
    if not os.path.exists(fp):
        f = open(fp,'a')
        f.write("#DataRange: "+str(YEARS)+"\n")
        f.write("#"+comment+"\n")
        f.write("date,time,oTemp,rain,dTemp,heatingOn\n")
    else:
        f = open(fp,'a')

    f.write(date+','+str(time)+','+str(oTemp)+','+str(rain)+','+str(dTemp)+','+str(on)+'\n')
    f.close()
    

def main():
    weatherRecords = readData(WD_FP)

    for index,row in weatherRecords.iterrows():
        date = datetime.strptime(row['date'],"%Y-%m-%d")
        desiredTemp,on = scheduleA.isWeekDay(date,row['time'],row['temp'],row['rain'],MIN_OUTSIDE_TEMP,MIN_HEATING_TEMP,NOISE_PROB,random.random())
        writeData(OUTPUT_FP,row['date'],row['time'],row['temp'],row['rain'],desiredTemp,on,COMMENT)


if __name__ == "__main__": main()

