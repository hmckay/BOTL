import numpy as np
import pandas as pd
import operator
from . import createModel
import datetime as dt
from sklearn import metrics
from scipy.linalg import norm

PROB_THRESHOLD = 0
ACC_THRESHOLD = 0
STABLE_THRESHOLD = 0
def newHistory(acc,prob,stable,model,startIDX):
    global ACC_THRESHOLD
    global PROB_THRESHOLD
    global STABLE_THRESHOLD
    ACC_THRESHOLD = acc
    PROB_THRESHOLD = prob
    STABLE_THRESHOLD = stable
    modelInfo = {'model':model,'usedFor':0}
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
    return stable

def getLongestStable(existingModels):
    stableID = max(existingModels.items(),key=lambda v: existingModels[v[0]]['usedFor'])[0]
    stable = existingModels[stableID]['model']
    return stable

def addNewModel(existingModels,transitionMatrix,model,prevModID,initTM):
    newID = len(existingModels)+1
    modelInfo = {'model':model,'usedFor':0}
    existingModels[newID]= modelInfo
    transitionMatrix[newID]={}
    
    if prevModID != 0 and prevModID != newID:
        transitionMatrix[prevModID][newID] = 1
    
    return newID,existingModels,transitionMatrix

def addRepeatModel(transitionMatrix,newModID,prevModID):
    if newModID in transitionMatrix[prevModID]:
        transitionMatrix[prevModID][newModID] += 1
    elif newModID != prevModID:
        transitionMatrix[prevModID][newModID] = 1
    return transitionMatrix

def searchModels(modelList,existingModels,df,target,DROP_FIELDS):
    acc = 0
    nextModel = 0
    for m in modelList:
        if existingModels[m]['usedFor'] >= STABLE_THRESHOLD or acc == 0:
            testDF = df.copy()
            mod = existingModels[m]['model']
            res = createModel.initialPredict(testDF,mod,target,DROP_FIELDS)
            thisAcc = metrics.r2_score(res[target],res['predictions'])
            if acc == 0:
                acc=thisAcc
                nextModel=m
            if thisAcc>acc:
                acc=thisAcc
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

def findStartingConcept(data,existingModels,transitionMatrix,startIDX,model,target,DROP_FIELDS):
    modelID = -1
    diff = 100
    for mID,info in existingModels.items():
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
    
    modelDetails = {'modelID':modelID,'startIDX':startIDX,'endIDX':None}

    return modelDetails,existingModels,transitionMatrix

def updateLastModelUsage(modelOrder,existingModels,endIDX):
    modelOrder[-1]['endIDX'] = endIDX
    lastModel = modelOrder[-1]
    indexDiff = 0
    if lastModel['endIDX'] != lastModel['startIDX']:
        indexDiff = lastModel['endIDX']-lastModel['startIDX']
    existingModels[lastModel['modelID']]['usedFor'] += indexDiff
    return modelOrder, existingModels


# def updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX):
    # modelOrder[-1]['endIDX'] = startIDX-1
def updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX,endLen):
    modelOrder[-1]['endIDX'] = startIDX-endLen
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

def reuseModels(existingModels,DF,target,DROP_FIELDS):
    df = DF.copy()
    acc,nextModel = searchModels(existingModels,existingModels,df,target,DROP_FIELDS)
    if acc<ACC_THRESHOLD:
        print("CANT reuse model "+str(nextModel)+": "+str(acc))
        return False
    else:
        print("can reuse model "+str(nextModel)+": "+str(acc))
        return True


def tempModel(existingModels,transitionMatrix,modelOrder,DF,currentModID,target,DROP_FIELDS,startIDX,initTM,tempModel):
    if not tempModel:
        # modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX)
        modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX,len(DF))
    df = DF.copy()
    m = createModel.createPipeline(df,target,DROP_FIELDS)
    nextModel = currentModID
    
    return nextModel,existingModels,transitionMatrix,modelOrder,m


def nextModels(existingModels,transitionMatrix,modelOrder,DF,currentModID,target,DROP_FIELDS,startIDX,initTM,tempModel):
    if not tempModel:
        # modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX)
        modelOrder, existingModels, transitionMatrix, currentModID = updateModelUsage(modelOrder,currentModID,existingModels,transitionMatrix,startIDX,len(DF))
    df = DF.copy()
    successors = transitionMatrix.get(currentModID).copy()
    total = sum(successors.values())
    if total == 0:
        acc,nextModel = searchModels(existingModels,existingModels,df,target,DROP_FIELDS)
        # print("model "+str(nextModel)+": "+str(acc))
        if acc<ACC_THRESHOLD:
            repeatModel = 0
            # print("generating new model becuase no successors")
            m = createModel.createPipeline(df,target,DROP_FIELDS)
            nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,m,currentModID,initTM)
        else:
            # print("historical-reactive")
            transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
        #print transitionMatrix

        orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
        modelOrder.append(orderDetails)
        return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['model']

    successors.update((x,y/total) for x,y in list(successors.items()))
    max_val = max(iter(successors.items()),key=operator.itemgetter(1))[1]
    nextModels = []
    nextModels = [k for k, v in successors.items() if v == max_val and max_val >= PROB_THRESHOLD]
    #print "next models: " + str(nextModels)
    nextModel = 0
    acc = 0
    repeatModel = 1
    print(ACC_THRESHOLD)
    if len(nextModels)>0:
        # nextModels.append(1)
        acc,nextModel = searchModels(nextModels,existingModels,df,target,DROP_FIELDS)
        # print("proactive acc: "+str(acc))
    # else:
    if acc < ACC_THRESHOLD:
        # print("historical-reactive")
        acc, nextModel = searchModels(existingModels,existingModels,df,target,DROP_FIELDS)
    
    if acc < ACC_THRESHOLD:
        #learn new concept
        #print "learn new model acc: "+str(acc)
        repeatModel = 0
        # print("generating new model")
        m = createModel.createPipeline(df,target,DROP_FIELDS)
        nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,m,currentModID,initTM)
    print(str(nextModel)+": "+str(acc))

    if repeatModel == 1:
        transitionMatrix = addRepeatModel(transitionMatrix,nextModel,currentModID)
    
    orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
    modelOrder.append(orderDetails)

    return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['model']
    # m = createModel.createPipeline(df,target,DROP_FIELDS)
    # nextModel, existingModels, transitionMatrix = addNewModel(existingModels,transitionMatrix,m,currentModID,initTM)
    # orderDetails = {'modelID':nextModel,'startIDX':startIDX,'endIDX':None}
    # modelOrder.append(orderDetails)

    # return nextModel,existingModels,transitionMatrix,modelOrder,existingModels[nextModel]['model']
