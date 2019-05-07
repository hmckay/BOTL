from __future__ import division
import numpy as np
import math

def noHeat(oTemp,threshold,baseTemp):
    if oTemp < threshold:
        return baseTemp,1
    return oTemp,0

def morning(oTemp,rain,a,t1,t2):
    if oTemp < t1:
        if rain:
            return 24+a, 1
        return 22+a, 1
    elif oTemp < t2:
        return 20+a,1
    
    return oTemp,0

def mid(oTemp,rain,a,t1,t2):
    if oTemp < t1:
        return 22+a,1
    elif oTemp < t2:
        if rain:
            return 21+a,1
        return 20+a,1
    return oTemp,0

def night(oTemp,rain,a,t1):
    if oTemp < t1:
        if rain:
            return 14+a,1
        return 12+a,1
    return oTemp,0

