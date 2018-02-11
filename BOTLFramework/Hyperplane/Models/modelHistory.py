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
def newHistory(acc,prob,stable,model,startDate):
    global ACC_THRESHOLD
    global PROB_THRESHOLD
    global STABLE_THRESHOLD
    ACC_THRESHOLD = acc
    PROB_THRESHOLD = prob
    STABLE_THRESHOLD = stable
    modelInfo = {'model':model,'daysOfUse':0}
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
    return stable

def sendToControl(modelID,model):
    pass

def addNewModel(existingModels,transitionMatrix,model,prevModID,initTM):
    newID = len(existingModels)+1
    modelInfo = {'model':model,'daysOfUse':0}
    existingModels[newID]= modelInfo
    transitionMatrix[newID]={}
    
    if prevModID != 0 and prevModID != newID:
        transitionMatrix[prevModID][newID] = 1
    
    return newID,existingModels,transitionMatrix

def addRepeatModel(transitionMatrix,newModID,prevModID):
    if transitionMatrix[prevModID].has_key(newModID):
        transitionMatrix[prevModID][newModID] += 1
    elif newModID != prevModID:
        transitionMatrix[prevModID][newModID] = 1
    return transitionMatrix

def searchModels(modelList,existingModels,df,target,DROP_FIELDS):
    acc = 0
    nextModel = 0
    for m in modelList:
        if existingModels[m]['daysOfUse'] >= STABLE_THRESHOLD:
            testDF = df.copy()
            mod = existingModels[m]['model']
            res = createModel.initialPredict(testDF,mod,target,DROP_FIELDS)
            r2 = metrics.r2_score(res[target],res['predictions'])
            if r2>acc:
                acc=r2
                nextModel=m
    return acc, nextModel


def compareModels(data,m1,m2,target,DROP_FIELDS):
    d1 = data.copy()
    d2 = data.copy()
    y1 = createModel.initialPredict(d1,m1,target,DROP_FIELDS)['predictions']
    y2 = createModel.initialPredict(d2,m2,target,DROP_FIELDS)['predictions']
   
    diffs = abs(y1-y2)
    maxDiff = max(diffs)
    return maxDiff

def findStartingConcept(data,existingModels,transitionMatrix,startDate,model,target,DROP_FIELDS):
    modelID = -1
    diff = 100
    for mID,info in existingModels.iteritems():
        if model == False:
            model = info['model']
            modelID = mID
        else:
            d2 = data.copy()
            tempM = info['model']
            compare = compareModels(d2,model,tempM,tempT,target,DROP_FIELDS)
            if compare < diff:
                diff = compare
                modelID = mID
                if diff == 0: break
    if modelID == -1:
        modelID, exitsingModels,transitionMatrix = addNewModel(existingModels,transitionMatrix,model,0,False)
    
    modelDetails = {'modelID':modelID,'startDate':startDate,'endDate':None}

    return modelDetails,existingModels,transitionMatrix

def updateLastModelUsage(modelOrder,existingModels,endDate):
    modelOrder[-1]['endDate'] = endDate
    lastModel = modelOrder[-1]
    dayDiff = 0
    if lastModel['endDate'] != lastModel['startDate']:
        dayDiff = lastModel['endDate']-lastModel['startDate']
    existingModels[lastModel['modelID']]['daysOfUse'] += dayDiff
    return modelOrder, existingModels


def updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startDate):
    modelOrder[-1]['endDate'] = startDate-1
    lastModel = modelOrder[-1]
    dayDiff = 0
    if lastModel['endDate'] != lastModel['startDate']:
        dayDiff = (lastModel['endDate']-lastModel['startDate'])
    existingModels[lastModel['modelID']]['daysOfUse'] += dayDiff

    if dayDiff < STABLE_THRESHOLD and len(modelOrder)>1:
        falseFrom = modelOrder[-2]['modelID']
        falseTo = modelOrder[-1]['modelID']
        if transitionMatrix[falseFrom].has_key(falseTo):
            if transitionMatrix[falseFrom][falseTo]>0:
                transitionMatrix[falseFrom][falseTo] -= 1
    back = -1
    while dayDiff < STABLE_THRESHOLD:
        if len(modelOrder) <= (back*-1):
            lastSuccess = modelOrder[back]
            currentModID = lastSuccess['modelID']
            break
        else:
            lastSuccess = modelOrder[back-1]
        currentModID = lastSuccess['modelID']
        dayDiff = (lastSuccess['endDate']-lastSuccess['startDate'])
        back -= 1
    return modelOrder,existingModels,transitionMatrix, currentModID


def nextModels(existingModels,transitionMatrix,modelOrder,DF,currentModID,target,DROP_FIELDS,startDate,initTM):
    modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startDate)
    df = DF.copy()
    successors = transitionMatrix.get(currentModID).copy()
    total = sum(successors.values())
    if total == 0:
        acc,nextModel = searchModels(existingModels,existingModels,df,target,DROP_FIELDS)
        if acc<ACC_THRESHOLD:
            repeatModel = 0
            m = createModel.createPipeline(df,target,DROP_FIELDS)
            nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,m,currentModID,initTM)
        else:
            transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)

        orderDetails = {'modelID':nextModel,'startDate':startDate,'endDate':None}
        modelOrder.append(orderDetails)
        return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['model']

    successors.update((x,y/total) for x,y in successors.items())
    max_val = max(successors.iteritems(),key=operator.itemgetter(1))[1]
    nextModels = []
    nextModels = [k for k, v in successors.iteritems() if v == max_val and max_val >= PROB_THRESHOLD]
    nextModel = 0
    acc = 0
    repeatModel = 1
    if len(nextModels)>0:
        nextModels.append(1)
        acc,nextModel = searchModels(nextModels,existingModels,df,target,DROP_FIELDS)
    else:
        acc, nextModel = searchModels(existingModels,existingModels,df,target,DROP_FIELDS)
    
    if acc < ACC_THRESHOLD:
        repeatModel = 0
        m = createModel.createPipeline(df,target,DROP_FIELDS)
        nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,m,currentModID,initTM)
    

    if repeatModel == 1:
        transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
    
    orderDetails = {'modelID':nextModel,'startDate':startDate,'endDate':None}
    modelOrder.append(orderDetails)

    return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['model']
