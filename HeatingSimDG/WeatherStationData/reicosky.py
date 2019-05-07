from __future__ import division
import os
import math
import numpy as np
import random
from random import gauss
from calendar import monthrange

script_dir = os.path.dirname(__file__)

def getHour(time):
    hour = int(time/100)
    if (time%100) == 30:
        hour+=0.5
    elif (time%100)>30:
        hour += 1
    return hour

def reicosky(sunrise,Tmax,Tmin,time):
    RISE = getHour(sunrise)
    HOUR = getHour(time)
    TAVE = (Tmin+Tmax)/2
    AMP = (Tmax-Tmin)/2
    Hprime = 0
    temp = 0

    if HOUR < RISE: #before sunrise
        Hprime = HOUR+10
    if HOUR > 14: #in evening
        Hprime = HOUR-14
    
    if(HOUR < RISE or HOUR > 14): #getting cooler
        temp = TAVE + AMP * (math.cos(math.pi*Hprime/(10+RISE)))
    else: #getting warmer
        temp = TAVE - AMP * (math.cos(math.pi*(HOUR-RISE)/(14-RISE)))

    return temp

def rainfall(rain,time):
    pass

def windspeed(meanS, maxS,time,prev):
    speed = np.random.normal(meanS,(maxS-meanS))
    threshold = 0.6
    upper = meanS
    lower = 0
    if prev < meanS:
        threshold = 0.6
        upper = meanS
        lower = 0
    else:
        threshold = 0.6
        upper = maxS
        lower = meanS
    p = random.random()
    if speed > upper and p>threshold:
        #speed = speed%maxS
        speed = speed%(meanS+((maxS+meanS)/2))
        #speed = speed%meanS
    if speed > upper:
        speed = lower
    #elif speed > meanS and p >0.5:
    #    speed = speed%(meanS+((maxS+meanS)/2))
    return speed

def dayData(date,sunrise,tempMax,tempMin):
    for time in range(100,2401,100):
        temperatureHalf = reicosky(sunrise,tempMax,tempMin,time-70)
        temperatureHour = reicosky(sunrise,tempMax,tempMin,time)
        print str(time-70) + ": "+str(temperatureHalf)
        print str(time) + ": "+str(temperatureHour)

'''
total = 0
itera = 1000
for i in range(0,1000):
    wind = windspeed(21,27,1100,0)
    total += wind
    if wind == 0: itera-=1
    print wind
print "total"
print total/itera
'''

