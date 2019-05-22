import numpy as np
import pandas as pd
import operator
from . import createModel
import datetime as dt
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.kernel_approximation import RBFSampler as RBF
from sklearn import metrics
from scipy.linalg import norm

PROB_THRESHOLD = 0
ACC_THRESHOLD = 0
STABLE_THRESHOLD = 0
SOURCE_MODELS = dict()
def newHistory(acc,prob,stable,sourceModels,targetModel,startIDX):
    global ACC_THRESHOLD
    global PROB_THRESHOLD
    global STABLE_THRESHOLD
    global SOURCE_MODEL
    ACC_THRESHOLD = acc
    PROB_THRESHOLD = prob
    STABLE_THRESHOLD = stable
    SOURCE_MODELS = sourceModels
    modelInfo = {'targetModel':targetModel,'usedFor':0}
    existingModels = dict()
    existingModels[1] = modelInfo
    transitionMatrix=dict()
    transitionMatrix[1] = {}
    orderDetails = {'modelID':1,'startIDX':startIDX,'endIDX':None}
    modelOrder = []
    modelOrder.append(orderDetails)
    return existingModels,transitionMatrix,1,modelOrder

def getStableModels(existingModels):
    stable = dict((k,v) for k,v in existingModels.items() if v['usedFor']>=STABLE_THRESHOLD)
    #print stable
    return stable

def addNewModel(existingModels,transitionMatrix,targetModel,prevModID,initTM):
    newID = len(existingModels)+1
    modelInfo = {'targetModel':targetModel,'usedFor':0}
    existingModels[newID]= modelInfo
    transitionMatrix[newID]={}
    
    if not prevModID == 0:
        transitionMatrix[prevModID][newID] = 1
    
    return newID,existingModels,transitionMatrix

def addRepeatModel(transitionMatrix,newModID,prevModID):
    if newModID in transitionMatrix[prevModID]:
        transitionMatrix[prevModID][newModID] += 1
    else:
        transitionMatrix[prevModID][newModID] = 1
    return transitionMatrix

def searchModels(modelList,existingModels,df,tLabel,DROP_FIELDS):
    acc = 0
    nextModel = 0
    for m in modelList:
        if existingModels[m]['usedFor'] >= STABLE_THRESHOLD or acc == 0:
            testDF = df.copy()
            mod = existingModels[m]['targetModel']
            res = createModel.initialPredict(testDF,mod,tLabel,DROP_FIELDS)
            thisAcc = metrics.r2_score(res[tLabel],res['predictions'])
            if acc == 0:
                acc=thisAcc
                nextModel=m
        
            if thisAcc>acc:
                acc=thisAcc
                nextModel=m

    return acc, nextModel


def compareModels(data,m1,m2,tLabel,DROP_FIELDS):
    d1 = data.copy()
    d2 = data.copy()
    y1 = createModel.initialPredict(d1,m1,tLabel,DROP_FIELDS)['predictions']
    y2 = createModel.initialPredict(d2,m2,tLabel,DROP_FIELDS)['predictions']
   
    diffs = abs(y1-y2)
    maxDiff = max(diffs)
    return maxDiff

def updateLastModelUsage(modelOrder,existingModels,endIDX):
    modelOrder[-1]['endIDX'] = endIDX
    lastModel = modelOrder[-1]
    indexDiff = 0
    if lastModel['endIDX'] != lastModel['startIDX']:
        indexDiff = (lastModel['endIDX']-lastModel['startIDX'])
    existingModels[lastModel['modelID']]['usedFor'] += indexDiff
    return modelOrder, existingModels


def updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX):
    modelOrder[-1]['endIDX'] = startIDX-1
    lastModel = modelOrder[-1]
    indexDiff = 0
    if lastModel['endIDX'] != lastModel['startIDX']:
        indexDiff = (lastModel['endIDX']-lastModel['startIDX'])
    existingModels[lastModel['modelID']]['usedFor'] += indexDiff

    if indexDiff < STABLE_THRESHOLD and len(modelOrder)>1:
        falseFrom = modelOrder[-2]['modelID']
        falseTo = modelOrder[-1]['modelID']
        if falseTo in transitionMatrix[falseFrom]:
            if transitionMatrix[falseFrom][falseTo]>0:
                transitionMatrix[falseFrom][falseTo] -= 1
    back = -1
    while indexDiff < STABLE_THRESHOLD:
        if len(modelOrder) <= (back*-1):
            lastSuccess = modelOrder[back]
            currentModID = lastSuccess['modelID']
            break
        else:
            lastSuccess = modelOrder[back-1]
        currentModID = lastSuccess['modelID']
        indexDiff = (lastSuccess['endIDX']-lastSuccess['startIDX'])
        back -= 1

    return modelOrder,existingModels,transitionMatrix, currentModID


def nextModels(existingModels,transitionMatrix,modelOrder,DF,currentModID,tLabel,DROP_FIELDS,startIDX,initTM):
    modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX)
    df = DF.copy()
    successors = transitionMatrix.get(currentModID).copy()
    total = sum(successors.values())
    if total == 0:
        acc,nextModel = searchModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
        if acc<ACC_THRESHOLD:
            repeatModel = 0
            #generating new model becuase no successors
            targetModel = createModel.createPipeline(df,tLabel,DROP_FIELDS)
            nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,targetModel,currentModID,initTM)
        else:
            #historical-reactive
            transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)

        orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
        modelOrder.append(orderDetails)
        return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['targetModel']

    successors.update((x,y/total) for x,y in list(successors.items()))
    max_val = max(iter(successors.items()),key=operator.itemgetter(1))[1]
    nextModels = []
    nextModels = [k for k, v in successors.items() if v == max_val and max_val >= PROB_THRESHOLD]

    nextModel = 0
    acc = 0
    repeatModel = 1
    if len(nextModels)>0:
        nextModels.append(1)
        acc,nextModel = searchModels(nextModels,existingModels,df,tLabel,DROP_FIELDS)
        #proactive 
    else:
        #historical-reactive
        acc, nextModel = searchModels(existingModels,existingModels,df,tLabel,DROP_FIELDS)
    
    
    if acc < ACC_THRESHOLD:
        #learn new concept
        repeatModel = 0
        targetModel = createModel.createPipeline(df,tLabel,DROP_FIELDS)
        nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,targetModel,currentModID,initTM)
    

    if repeatModel == 1:
        transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
    
    orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
    modelOrder.append(orderDetails)

    return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['targetModel']


