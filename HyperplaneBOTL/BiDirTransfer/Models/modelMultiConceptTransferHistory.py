import numpy as np
import pandas as pd
import operator
import createModel
import datetime as dt
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.kernel_approximation import RBFSampler as RBF
from sklearn import metrics
from scipy.linalg import norm

PROB_THRESHOLD = 0
ACC_THRESHOLD = 0
STABLE_THRESHOLD = 0
SOURCE_MODELS = dict()

def newHistory(acc,prob,stable,sourceModels,targetModel,startDate):
    global ACC_THRESHOLD
    global PROB_THRESHOLD
    global STABLE_THRESHOLD
    global SOURCE_MODELS
    ACC_THRESHOLD = acc
    PROB_THRESHOLD = prob
    STABLE_THRESHOLD = stable

    #add communication - see if controller has any models to send to us


    SOURCE_MODELS = sourceModels
    modelInfo = {'targetModel':targetModel,'daysOfUse':0}
    existingModels = dict()
    existingModels[1] = modelInfo
    transitionMatrix=dict()
    transitionMatrix[1] = {}
    orderDetails = {'modelID':1,'startDate':startDate,'endDate':None}
    modelOrder = []
    modelOrder.append(orderDetails)
    return existingModels,transitionMatrix,1,modelOrder

def getStableModels(existingModels):
    stable = dict((k,v) for k,v in existingModels.iteritems() if v['daysOfUse']>=STABLE_THRESHOLD)
    #print stable
    return stable

def addSourceModel(newModels):
    global SOURCE_MODELS
    for modID,mod in newMod.iteritems():
        SOURCE_MODELS[modID] = mod
    return SOURCE_MODELS


def addNewModel(existingModels,transitionMatrix,targetModel,prevModID,initTM):
    newID = len(existingModels)+1
    modelInfo = {'targetModel':targetModel,'daysOfUse':0}
    existingModels[newID]= modelInfo
    transitionMatrix[newID]={}
    
    if not prevModID == 0:
        transitionMatrix[prevModID][newID] = 1
    #print "new model: "+str(prevModID)+" - "+str(newID)

    
    return newID,existingModels,transitionMatrix

def addRepeatModel(transitionMatrix,newModID,prevModID):
    if transitionMatrix[prevModID].has_key(newModID):
        transitionMatrix[prevModID][newModID] += 1
    else:
        transitionMatrix[prevModID][newModID] = 1
    #print "repeat model: "+str(prevModID)+" - "+str(newModID)
    return transitionMatrix

def searchModels(modelList,existingModels,df,tLabel,DROP_FIELDS):
    acc = 0
    nextModel = 0
    for m in modelList:
        if existingModels[m]['daysOfUse'] >= STABLE_THRESHOLD:
            testDF = df.copy()
            mod = existingModels[m]['targetModel']
            res = createModel.initialPredict(testDF,mod,tLabel,DROP_FIELDS)
            r2 = metrics.r2_score(res[tLabel],res['predictions'])
        
            #print r2
            if r2>acc:
                acc=r2
                nextModel=m
    return acc, nextModel


def compareModels(data,m1,m2,tLabel,DROP_FIELDS):
    d1 = data.copy()
    d2 = data.copy()
    y1 = createModel.initialPredict(d1,m1,tLabel,DROP_FIELDS)['predictions']
    y2 = createModel.initialPredict(d2,m2,tLabel,DROP_FIELDS)['predictions']
   
    diffs = abs(y1-y2)
    maxDiff = max(diffs)
    #hellinger = norm(np.sqrt(y1['predictions'])-np.sqrt(y2['predictions']))/np.sqrt(2)
    #print hellinger
    return maxDiff



def updateLastModelUsage(modelOrder,existingModels,endDate):
    modelOrder[-1]['endDate'] = endDate
    #print modelOrder[-1]['endDate']
    lastModel = modelOrder[-1]
    dayDiff = 0
    if lastModel['endDate'] != lastModel['startDate']:
        dayDiff = (lastModel['endDate']-lastModel['startDate'])
    #print dayDiff,STABLE_THRESHOLD
    existingModels[lastModel['modelID']]['daysOfUse'] += dayDiff
    return modelOrder, existingModels


def updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startDate):
    #print startDate
    modelOrder[-1]['endDate'] = startDate-1
    #print modelOrder[-1]['endDate']
    lastModel = modelOrder[-1]
    dayDiff = 0
    if lastModel['endDate'] != lastModel['startDate']:
        dayDiff = (lastModel['endDate']-lastModel['startDate'])
    #dayDiff = (lastModel['endDate']-lastModel['startDate']).days
    #print dayDiff,STABLE_THRESHOLD
    existingModels[lastModel['modelID']]['daysOfUse'] += dayDiff

    if dayDiff < STABLE_THRESHOLD and len(modelOrder)>1:
        falseFrom = modelOrder[-2]['modelID']
        falseTo = modelOrder[-1]['modelID']
        if transitionMatrix[falseFrom].has_key(falseTo):
            if transitionMatrix[falseFrom][falseTo]>0:
                transitionMatrix[falseFrom][falseTo] -= 1
    back = -1
    while dayDiff < STABLE_THRESHOLD:
        #print dayDiff, STABLE_THRESHOLD
        if len(modelOrder) <= (back*-1):
            lastSuccess = modelOrder[back]
            currentModID = lastSuccess['modelID']
            break
        else:
            lastSuccess = modelOrder[back-1]
        #lastSuccess = modelOrder[back-1]
        currentModID = lastSuccess['modelID']
        dayDiff = (lastSuccess['endDate']-lastSuccess['startDate'])
        back -= 1

    return modelOrder,existingModels,transitionMatrix, currentModID

def sendToControl(modelID,model):
    pass

def nextModels(existingModels,transitionMatrix,modelOrder,DF,currentModID,tLabel,DROP_FIELDS,startDate,initTM):
    modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startDate)
    df = DF.copy()

    successors = transitionMatrix.get(currentModID).copy()
    total = sum(successors.values())
    if total == 0:
        acc,nextModel = searchModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
        if acc<ACC_THRESHOLD:
            repeatModel = 0
            #print "generating new model becuase no successors"
            targetModel = createModel.createPipeline(df,tLabel,DROP_FIELDS)
            nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,targetModel,currentModID,initTM)
            #SEND SOURCE MODEL TO CONTROLLER
            sendToControl(nextModel,targetModel)
        else:
            #print "historical-reactive"
            transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
        #print transitionMatrix

        orderDetails = {'modelID':nextModel,'startDate':startDate,'endDate':None}
        modelOrder.append(orderDetails)
        return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['targetModel']

    successors.update((x,y/total) for x,y in successors.items())
    max_val = max(successors.iteritems(),key=operator.itemgetter(1))[1]
    nextModels = []
    nextModels = [k for k, v in successors.iteritems() if v == max_val and max_val >= PROB_THRESHOLD]

    #print "next models: " + str(nextModels)
    nextModel = 0
    acc = 0
    repeatModel = 1
    '''
    if len(nextModels) == 1:
        nextModel = nextModels[0]
        acc = 
        print "proactive - should be for 2nd model"
    '''
    if len(nextModels)>0:
        nextModels.append(1)
        acc,nextModel = searchModels(nextModels,existingModels,df,tLabel,DROP_FIELDS)
        #print "proactive acc: "+str(acc)
    else:
        #print "historical-reactive"
        acc, nextModel = searchModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
    
    
    if acc < ACC_THRESHOLD:
        #learn new concept
        #print "learn new model acc: "+str(acc)
        repeatModel = 0
        #print "generating new model"
        targetModel = createModel.createPipeline(df,tLabel,DROP_FIELDS)
        nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,targetModel,currentModID,initTM)
        #SEND SOURCE MODEL TO CONTROLLER
        sendToControl(nextModel,targetModel)
    

    if repeatModel == 1:
        #print "reusing model " +str(nextModel)
        transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
    
    orderDetails = {'modelID':nextModel,'startDate':startDate,'endDate':None}
    modelOrder.append(orderDetails)

    #res = createModel.initialPredict(df.copy(),existingModels[nextModel]['transformer'],existingModels[nextModel]['model'],target,DROP_FIELDS)
    #acc = metrics.r2_score(res[target],res['predictions'])
    #print transitionMatrix
    #print acc
    #print "nextModel to use: "+str(nextModel)
    return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['targetModel']


