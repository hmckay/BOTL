from __future__ import division
import sys
import numpy as np
import pandas as pd
import time
import socket
import pickle
import time

from Models import createModel,modelHistory
from Models import modelMultiConceptTransfer as modelMulti
from Models import modelMultiConceptTransferHistory as modelMultiHistory
import preprocessData as preprocess
from datetime import datetime,timedelta
from sklearn import metrics


DROP_FIELDS = ['predictions']
tLabel='y'

INIT_DAYS = 0
MODEL_HIST_THRESHOLD_ACC = 0
MODEL_HIST_THRESHOLD_PROB = 0
STABLE_SIZE = 0
MAX_WINDOW = 0
THRESHOLD = 0
PORT = 0
SIM_FROM = 0
SIM_TO = 0
FP = ''
modelsSent = []
sentFlag = 0


def calcError(y,preds):
    y= np.round(y,decimals=1)
    return metrics.r2_score(y,preds)

def modelReadyToSend(modelID,model,s):
    successFlag = 0
    if modelID not in modelsSent:
        successFlag = handshake(modelID,model,s)
    else:
        print "model already sent"
        return 1

    if successFlag:
        modelsSent.append(modelID)
        print "sucessfully sent model"
        return 1
    else:
        print "unsucessful send"
        return 0

def handshake(modelID,model,s):
    print modelID
    modelToSend = pickle.dumps(model)
    brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
    numPackets = len(brokenBytes)
    RTSmsg = 'RTS,'+str(modelID)+','+str(numPackets)
    s.sendall(RTSmsg)
    ack = s.recv(1024)
    ackNumPackets = int(ack.split(',')[1])

    if ackNumPackets == numPackets:
        return sendModel(modelID,brokenBytes,s)
    return 0

def sendModel(modelID,brokenBytes,s):
    for i in brokenBytes:
        s.sendall(i)
        print "finished sending"
    recACK = s.recv(1024)
    if modelID == int(recACK.split(',')[1]):
        return 1
    return 0

def readyToReceive(s,sourceModels):
    s.sendall('RTR,'+str(ID))
    data = s.recv(1024)
    print "TARGET: num models to receive "+repr(data)
    ACKFlag = data.split(',')[0]
    numModels = int(data.split(',')[1])
    
    if ACKFlag == 'ACK':
        if numModels == 0:
            s.sendall('END')
            return sourceModels
        s.sendall('ACK')
        for i in range(0,numModels):
            sourceModels = receiveModels(s,sourceModels)
    return sourceModels

def receiveModels(s,sourceModels):
    RTSInfo = s.recv(1024)
    RTSFlag = RTSInfo.split(',')[0]
    sourceModID = RTSInfo.split(',')[1]
    numPackets = int(RTSInfo.split(',')[2])
    s.sendall('ACK,'+str(numPackets))
    
    pickledModel = ""
    for i in range(0,numPackets):
        pickledModel = pickledModel + s.recv(1024)
    s.sendall('ACK,'+str(sourceModID))

    return storeSourceModel(sourceModID,pickledModel,sourceModels)

def storeSourceModel(sourceModID,pickledModel,sourceModels):
    model = pickle.loads(pickledModel)
    sourceModels[sourceModID] = model
    print model
    return sourceModels





def runRePro(df,sourceModels,tLabel,weightType,s):
    buildTMod = False
    sourceAlpha = 1
    targetModel = None
    existingModels = dict()
    transitionMatrix = dict()
    multiWeights = []
    weight = dict()
    weights = dict()
    for m in sourceModels:
        weight[m] = 1.0/len(sourceModels)
    print "initial weights:"
    print weight
    weights['sourceR2s']=weight
    weights['totalR2']=len(sourceModels)
    multiWeights.append(weights)
    week = 0
    drifts = dict()
    storeFlag = False
    modelID = 0
    lastModID = 0
    modelCount = 0
    modelOrder = []
    windowSize = 0
    conceptSize = 0
    ofUse = 0
    sentFlag = 0
    window = pd.DataFrame(columns = df.columns)
    historyData = pd.DataFrame(columns = df.columns)
    startDate = df.index.min()
    startDoW = df.index.min()%20

    modelStart = []
    modelStart.append(startDate)
    p = pd.DataFrame(columns = df.columns)

    for idx in df.index:
        conceptSize+=1
        if idx%20 == startDoW and storeFlag == False:
            storeFlag = True
            week += 1
        
        if ofUse >= STABLE_SIZE and not sentFlag:
            print "trying to send "+str(modelID)
            sentFlag = modelReadyToSend(modelID,targetModel,s)
        if len(historyData) < MAX_WINDOW:
            if len(historyData)%5 == 4 and len(sourceModels)>0:
                weight = modelMulti.updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,weightType)
                print weight
            if not sourceModels:
                prediction = modelMulti.defaultInstancePredict(df,idx)
            else:
                prediction = modelMulti.initialInstancePredict(df,idx,sourceModels,weight,tLabel,DROP_FIELDS,weightType)
            historyData = historyData.append(prediction)
            p = p.append(prediction)

        elif (len(historyData) == MAX_WINDOW) and not buildTMod:
            if weightType != 'OLS' and weightType != 'OLSFE' and weightType != 'Ridge' and weightType != 'NNLS':
                weights['sourceR2s']=weight
                weights['totalR2']=sum(weight.itervalues())
            else: 
                weights = weight

            multiWeights.append(weights)
            buildTMod = True
            print "building target model: "+str(idx)
            sourceModels = readyToReceive(s,sourceModels)
            targetModel = createModel.createPipeline(historyData,tLabel,DROP_FIELDS)

            weights = modelMulti.calcWeights(historyData,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
            print "first prediction with target model: "+str(idx)
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
            
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            window = p.copy()

            print "first alpha value is: " +str(weights)

            ##########CREATE NEW HISTORY FOR ENSEMBLE WEIGHTS
            existingModels,transitionMatrix,modelID,modelOrder = modelMultiHistory.newHistory(MODEL_HIST_THRESHOLD_ACC,
                    MODEL_HIST_THRESHOLD_PROB,STABLE_SIZE,sourceModels,targetModel,startDate)
            drifts[modelCount] = week
            modelCount = 1
            
        #continue with repro
        else:
            if idx%20 == 0 and (weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'Ridge'):
                weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
            ofUse += 1
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
            historyData = historyData.append(prediction)
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            p = p.append(targetPred) 
            if idx%50 == 0:
                print idx, weights
            diff = abs(targetPred['predictions']-targetPred[tLabel])

            if diff >0.01 and windowSize ==0:
                window = window.append(targetPred)
                windowSize = len(window)
            elif windowSize>0:
                window = window.append(targetPred)
                windowSize = len(window)
                if windowSize == MAX_WINDOW:
                    err = calcError(window[tLabel],window['predictions'])
                    if err < THRESHOLD:
                        lastModID = modelID
                        modelCount+=1
                        startDate = idx
                        modelStart.append(startDate)
                        conceptSize = len(window)
                    
                        partial = window.copy()
                        #CALC ENSEMBLE WEIGHT
                        sourceModels=readyToReceive(s,sourceModels)
                        modelID, existingModels,transitionMatrix,modelOrder,targetModel = modelMultiHistory.nextModels(existingModels,transitionMatrix, 
                                modelOrder,partial,modelID,tLabel,DROP_FIELDS,startDate,False)
                        # send to controller
                        ofUse = 0
                        sentFlag = 0
                        weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
                        window = createModel.initialPredict(window,targetModel,tLabel,DROP_FIELDS)
                        if lastModID < modelID:
                            drifts[modelCount] = week
            if windowSize>=MAX_WINDOW:
                top = window.loc[window.index.min()]
                diffTop = abs(top['predictions']-top[tLabel])

                while (windowSize>=MAX_WINDOW or diffTop <=0.01):
                    window = window.drop([window.index.min()],axis=0)
                    windowSize = len(window)
                    if windowSize == 0:
                        break
                    top = window.loc[window.index.min()]
                    diffTop = abs(top['predictions']-top[tLabel])

        if idx%20 == (startDoW-1) and storeFlag == True:
            storeFlag = False
            w = dict()
            for k in weights['sourceR2s']:
                w[k] = weights['sourceR2s'][k]/weights['totalR2']
            multiWeights.append(w)
    endDate = historyData.index.max()
    modelOrder,existingModels = modelMultiHistory.updateLastModelUsage(modelOrder,existingModels,endDate)
    print "number concepts: "+str(modelCount)
    print modelStart
    print modelOrder
    print existingModels
    print transitionMatrix
    mse = metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])
    print mse
    print calcError(historyData[tLabel],historyData['predictions'])
    w = dict()
    for k in weights['sourceR2s']:
        w[k] = weights['sourceR2s'][k]/weights['totalR2']
    multiWeights.append(w)
    print "all weights are:"
    print multiWeights
    print metrics.r2_score(p[tLabel],p['predictions'])


    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights



def simTarget(weightType,s):
    sourceModels = dict()
    sourceModels = readyToReceive(s,sourceModels)

    print "source Models:"
    print sourceModels
    df = preprocess.pullData(FP)
    df = preprocess.subsetData(df,SIM_FROM,SIM_TO)
    df['predictions'] = np.nan
    
    historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights = runRePro(df,sourceModels,tLabel,weightType,s)
    print "======================"
    print "OVERALL PERFORMANCE"
    print "R2: " + str(metrics.r2_score(historyData[tLabel],historyData['predictions']))
    print "RMSE: "+str(np.sqrt(metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])))
    print "======================"
    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights

def main():
    global ID
    global PORT
    global SIM_FROM
    global SIM_TO
    global FP
    global INIT_DAYS
    global MODEL_HIST_THRESHOLD_ACC
    global MODEL_HIST_THRESHOLD_PROB
    global STABLE_SIZE
    global MAX_WINDOW
    global THRESHOLD

    HOST = 'localhost'
    ID = str(sys.argv[1])
    PORT = int(sys.argv[2])
    SIM_FROM = int(sys.argv[3])
    SIM_TO = int(sys.argv[4])
    FP = str(sys.argv[5])
    INIT_DAYS = int(sys.argv[6])
    MODEL_HIST_THRESHOLD_ACC = float(sys.argv[7])
    MODEL_HIST_THRESHOLD_PROB = float(sys.argv[8])
    STABLE_SIZE = int(sys.argv[9])
    MAX_WINDOW = int(sys.argv[10])
    THRESHOLD = float(sys.argv[11])
    weightType = str(sys.argv[12])

    '''
    # set up connection to controller #
    '''

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    s.sendall(ID)
    print "connection established"
    print ID, PORT, SIM_FROM, SIM_TO, FP, INIT_DAYS, MODEL_HIST_THRESHOLD_ACC, MODEL_HIST_THRESHOLD_PROB, STABLE_SIZE, MAX_WINDOW, THRESHOLD

    ack = s.recv(1024)
    print repr(ack)
    simTarget(weightType,s)

    s.close()

if __name__ == '__main__':main()

