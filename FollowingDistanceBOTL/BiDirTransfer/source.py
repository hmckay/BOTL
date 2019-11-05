
import sys
import numpy as np
import pandas as pd
import time
import socket
import pickle
import time
import os.path

from Models import createModel,modelHistory
from Models import modelMultiConceptTransfer as modelMulti
from Models import modelMultiConceptTransferHistory as modelMultiHistory
import preprocessData as preprocess
from datetime import datetime,timedelta
from sklearn import metrics
from scipy.stats import pearsonr



DROP_FIELDS = ['Time','predictions']
tLabel='FollowingTime'

INIT_DAYS = 0
MODEL_HIST_THRESHOLD_ACC = 0
MODEL_HIST_THRESHOLD_PROB = 0
STABLE_SIZE = 0
MAX_WINDOW = 0
THRESHOLD = 0
PORT = 0
SIM_FROM = 0
SIM_TO = 0
RUNID=0
NUMSTREAMS = 0
WEIGHTTYPE = ''
CULLTHRESH = 0
MITHRESH = 0
FP = ''
modelsSent = []
sentFlag = 0
UID = ''


def calcError(y,preds):
    return metrics.r2_score(y,preds)

def modelReadyToSend(modelID,model,s):
    successFlag = 0
    if modelID not in modelsSent:
        successFlag = handshake(modelID,model,s)
    else:
        print("model already sent")
        return 1

    if successFlag:
        modelsSent.append(modelID)
        print("sucessfully sent model")
        return 1
    else:
        print("unsucessful send")
        return 0

def handshake(modelID,model,s):
    print(modelID)
    modelToSend = pickle.dumps(model)
    lenofModel = len(modelToSend)
    brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
    numPackets = len(brokenBytes)
    RTSmsg = 'RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel)
    s.sendall(RTSmsg.encode())
    ack = s.recv(1024).decode()
    ackNumPackets = int(ack.split(',')[1])

    if ackNumPackets == numPackets:
        return sendModel(modelID,brokenBytes,s)
    return 0

def sendModel(modelID,brokenBytes,s):
    for i in brokenBytes:
        s.sendall(i)
        print("finished sending")
    recACK = s.recv(1024).decode()
    if modelID == int(recACK.split(',')[1]):
        return 1
    return 0

def readyToReceive(s,sourceModels):
    s.sendall(('RTR,'+str(ID)).encode())
    data = s.recv(1024).decode()
    print("TARGET: num models to receive "+repr(data))
    ACKFlag = data.split(',')[0]
    numModels = int(data.split(',')[1])
    print("ready to recieve function called")
    if ACKFlag == 'ACK':
        if numModels == 0:
            s.sendall(('END').encode())
            return sourceModels
        s.sendall(('ACK').encode())
        for i in range(0,numModels):
            sourceModels = receiveModels(s,sourceModels)
    return sourceModels

def receiveModels(s,sourceModels):
    RTSInfo = s.recv(1024).decode()
    RTSFlag = RTSInfo.split(',')[0]
    sourceModID = RTSInfo.split(',')[1]
    numPackets = int(RTSInfo.split(',')[2])
    lenofModel = int(RTSInfo.split(',')[3])
    print("NUMBER OF PACKETS EXPECTING: "+str(numPackets))
    s.sendall(('ACK,'+str(numPackets)).encode())
    
    pickledModel = b''
    while (len(pickledModel) < lenofModel):
        pickledModel = pickledModel+s.recv(1024)
    s.sendall(('ACK,'+str(sourceModID)).encode())

    return storeSourceModel(sourceModID,pickledModel,sourceModels)

def storeSourceModel(sourceModID,pickledModel,sourceModels):
    print("picked model len is: "+str(len(pickledModel)))
    model = pickle.loads(pickledModel)
    sourceModels[sourceModID] = model
    print(model['model'])
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
    print("initial weights:")
    print(weight)
    weights['sourceR2s']=weight
    weights['totalR2']=len(sourceModels)
    multiWeights.append(weights)
    numModelsUsed = []
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
    startIDX = df.index.min()+INIT_DAYS+1
    # startDoW = df.index.min()%20
    startDoW = df.index.min()%INIT_DAYS

    modelStart = []
    modelStart.append(startIDX)
    p = pd.DataFrame(columns = df.columns)

    for idx in df.index:
        conceptSize+=1
        if idx%INIT_DAYS == startDoW and storeFlag == False:
            storeFlag = True
            week += 1
        
        # if ofUse >= STABLE_SIZE and not sentFlag:
        # if (modelMultiHistory.getLenUsedFor(modelID,existingModels)) and (modelID not in modelsSent):
        if (ofUse + modelMultiHistory.getLenUsedFor(modelID,existingModels)) >= STABLE_SIZE and (modelID not in modelsSent):
            print("trying to send "+str(modelID))
            if 'OLSKPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData[(-1*STABLE_SIZE):],DROP_FIELDS,True)
                PCs = modelMultiHistory.getPCs(historyData[(-1*MAX_WINDOW):],DROP_FIELDS,True)
            elif 'OLSPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData[(-1*STABLE_SIZE):],DROP_FIELDS,False)
                PCs = modelMultiHistory.getPCs(historyData[(-1*MAX_WINDOW):],DROP_FIELDS,False)
            else:
                PCs = None
            existingModels[modelID]['PCs'] = PCs
            modelInfo = {'model':targetModel,'PCs':PCs}
            sentFlag = modelReadyToSend(modelID,modelInfo,s)
        
        #first window of data
        if len(historyData) < MAX_WINDOW:
            if (len(historyData)%(int(MAX_WINDOW/4)) == (int(MAX_WINDOW/4)-1)) and len(sourceModels)>0:
                weight = modelMulti.updateInitialWeights(historyData,sourceModels,tLabel,
                        DROP_FIELDS,weightType,CULLTHRESH,MITHRESH)
            if not sourceModels:
                prediction = modelMulti.defaultInstancePredict(df,idx)
                targetPred = prediction
            else:
                #"making prediction, source Models: "+str(sourceModels)
                prediction = modelMulti.initialInstancePredict(df,idx,sourceModels,weight,tLabel,
                        DROP_FIELDS,weightType,CULLTHRESH,MITHRESH)
                targetPred = modelMulti.defaultInstancePredict(df,idx)
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            if 'OLS' in weightType:
                if not sourceModels:
                    numModels = 0
                else:
                    modelWeights = weights['sourceR2s']
                    numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                    numModelsUsed.append(numModels)


            #    # resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
            #    # wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # if not sourceModels:
            #        # numModels = 0
            #    # else:
            #        # modelWeights = weights['sourceR2s']
            #        # numModels = sum(1 for val in modelWeights.values() if val != 0)
            #    # resultFile.write(str(idx)+','+str(wErr)+','+str(numModels)+'\n')
            #    # resultFile.close()
        #one window of data received
        elif (len(historyData) == MAX_WINDOW) and not buildTMod:
            # if (weightType != 'OLS' and weightType != 'OLSFE' and weightType != 'Ridge' and 
                    # weightType != 'NNLS' and weightType != 'OLSFEMI' and weightType != 'OLSCL'):
            if ('OLS' not in weightType and weightType != 'Ridge' and weightType != 'NNLS'):
                weights['sourceR2s']=weight
                weights['totalR2']=sum(weight.values())
            else: 
                weights = weight
            multiWeights.append(weights)
            #print "building first target model"
            buildTMod = True
            print("building target model: "+str(idx))
            #receive models from controller
            sourceModels = readyToReceive(s,sourceModels)
            targetModel = createModel.createPipeline(historyData,tLabel,DROP_FIELDS)
            PCs = None
            if 'OLSKPAC' in weightType:
                PCs = modelMultiHistory.getPCs(historyData,DROP_FIELDS,True)
            elif 'OLSPAC' in weightType:
                PCs = modelMultiHistory.getPCs(historyData,DROP_FIELDS,False)
            else:
                PCs = None

            weights = modelMulti.calcWeights(historyData,sourceModels,targetModel,tLabel,
                    DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,None,None,PCs)
            # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            if 'OLS' in weightType:
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            # if weightType == 'OLSFE' or weightType =='OLS':
            #    # resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
            #    # wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # modelWeights = weights['sourceR2s']
            #    # numModels = sum(1 for val in modelWeights.values() if val != 0)
            #    # resultFile.write(str(idx)+','+str(wErr)+','+str(numModels+1)+'\n')
            #    # resultFile.close()
            print("first prediction with target model: "+str(idx))
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType,None,None,PCs)
            ofUse+=1
            
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            
            #calculate ensemble weights - return dict

            print("first alpha value is: " +str(weights))

            ##########CREATE NEW HISTORY FOR ENSEMBLE WEIGHTS
            existingModels,transitionMatrix,modelID,modelOrder = modelMultiHistory.newHistory(MODEL_HIST_THRESHOLD_ACC,
                    MODEL_HIST_THRESHOLD_PROB,STABLE_SIZE,sourceModels,targetModel,startIDX,PCs)
            drifts[modelCount] = week
            modelCount = 1
            
        #continue with repro
        else:
            if weightType == 'OLSCL2' or 'OLSPAC' in weightType or 'OLSKPAC' in weightType:
                stableLocals = modelMultiHistory.getStableModels(existingModels)
            else:
                stableLocals = None
            if 'OLSPAC' in weightType or 'OLSKPAC' in weightType:
                PCs = existingModels[modelID]['PCs']
            else:
                PCs = None
            # if idx%25 == 0 and (weightType == 'OLS' or weightType == 'OLSFE' or 
                    # weightType == 'OLSFEMI' or weightType == 'Ridge' or weightType == 'OLSCL'):
            if idx%(int(MAX_WINDOW/2)) == 0 and ('OLS' in weightType or weightType == 'Ridge'):
                # stableLocals = modelMultiHistory.getStableModels(existingModels)
                weights = modelMulti.calcWeights(historyData[(-1*STABLE_SIZE):],sourceModels,
                        targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,PCs)
            ofUse += 1
            # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            if 'OLS' in weightType:
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            # if weightType == 'OLSFE' or weightType =='OLS':
            #    # resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
            #    # #wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # if len(historyData)<30:
            #        # wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            #    # else:
            #        # wErr = metrics.r2_score(historyData.iloc[(idx-31):][tLabel],historyData.iloc[(idx-31):]['predictions'])
            #    # modelWeights = weights['sourceR2s']
            #    # numModels = sum(1 for val in modelWeights.values() if val != 0)
            #    # resultFile.write(str(idx)+','+str(wErr)+','+str(numModels+1)+'\n')
            #    # resultFile.close()
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,
                    weightType,stableLocals,modelID,PCs)
            historyData = historyData.append(prediction)
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            p = p.append(targetPred) 
            if idx%50 == 0:
                print(idx, weights)
            diff = abs(targetPred['predictions']-targetPred[tLabel])

            if diff >0.1 and windowSize ==0:
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
                        startIDX = idx
                        modelStart.append(startIDX)
                        conceptSize = len(window)
                    
                        partial = window.copy()
                        '''
                            # need to check controller for available models, copy them to sourceModels
                            # handshake
                            # finally update source models to be used
                            # sourceModels = modelMultiHistory.addSourceModel(newModelsDict) # where (modelID,newModel)
                        '''
                        #CALC ENSEMBLE WEIGHT
                        sourceModels=readyToReceive(s,sourceModels)
                        modelID, existingModels,transitionMatrix,modelOrder,targetModel = modelMultiHistory.nextModels(existingModels,
                                transitionMatrix,modelOrder,partial,modelID,tLabel,DROP_FIELDS,startIDX,False,weightType)
                        # send to controller
                        ofUse = 0
                        sentFlag = 0
                        # stableLocals = modelMultiHistory.getStableModels(existingModels)
                        weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,
                                DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                        window = createModel.initialPredict(window,targetModel,tLabel,DROP_FIELDS)
                        if lastModID < modelID:
                            drifts[modelCount] = week
            if windowSize>=MAX_WINDOW:
                top = window.loc[window.index.min()]
                diffTop = abs(top['predictions']-top[tLabel])

                while (windowSize>=MAX_WINDOW or diffTop <=0.1):
                    window = window.drop([window.index.min()],axis=0)
                    windowSize = len(window)
                    if windowSize == 0:
                        break
                    top = window.loc[window.index.min()]
                    diffTop = abs(top['predictions']-top[tLabel])

        # if idx%75 == (startDoW-1) and storeFlag == True:
        if idx%(int(MAX_WINDOW/2)) == (startDoW-1) and storeFlag == True:
            storeFlag = False
            w = dict()
            for k in weights['sourceR2s']:
                w[k] = weights['sourceR2s'][k]/weights['totalR2']
            multiWeights.append(w)
    endIDX = historyData.index.max()
    modelOrder,existingModels = modelMultiHistory.updateLastModelUsage(modelOrder,existingModels,endIDX)
    print("number concepts: "+str(modelCount))
    print(modelStart)
    print(modelOrder)
    # print(existingModels)
    print(transitionMatrix)
    mse = metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])
    print(mse)
    print(calcError(historyData[tLabel],historyData['predictions']))
    w = dict()
    for k in weights['sourceR2s']:
        w[k] = weights['sourceR2s'][k]/weights['totalR2']
    multiWeights.append(w)
    print("all weights are:")
    print(multiWeights)
    print(metrics.r2_score(p[tLabel],p['predictions']))


    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed



def simTarget(weightType,s):
    sourceModels = dict()
    sourceModels = readyToReceive(s,sourceModels)

    #get models from controller
    print("source Models:")
    print(sourceModels)
    df = preprocess.pullData(FP)
    df = preprocess.subsetData(df,SIM_FROM,SIM_TO)
    df['predictions'] = np.nan
    
    historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed = runRePro(df,sourceModels,tLabel,weightType,s)
    numModelsUsedArr = np.array(numModelsUsed)
    meanNumModels = numModelsUsedArr.mean()
    print("model hist acc: "+str(MODEL_HIST_THRESHOLD_ACC))
    print("model hist prob: "+str(MODEL_HIST_THRESHOLD_PROB))
    print("stablesixe: "+str(STABLE_SIZE))
    print(" maxwindow: "+str(MAX_WINDOW))
    print("threshold: "+str(THRESHOLD))
    if WEIGHTTYPE == 'OLS':
        fileWeight = 'OLS'
    elif WEIGHTTYPE == 'OLSFEMI':
        fileWeight = str(WEIGHTTYPE)+str(MITHRESH)+"_"+str(CULLTHRESH)
    elif WEIGHTTYPE == 'OLSCL':
        fileWeight = 'OLSCL'
    elif WEIGHTTYPE == 'OLSCL2':
        fileWeight = 'OLSCL2'
    elif WEIGHTTYPE == 'OLSPAC':
        fileWeight = 'OLSPAC'
    elif WEIGHTTYPE == 'OLSPAC2':
        fileWeight = 'OLSPAC2'
    elif WEIGHTTYPE == 'OLSKPAC':
        fileWeight = 'OLSKPAC'
    elif WEIGHTTYPE == 'OLSKPAC2':
        fileWeight = 'OLSKPAC2'
    else:
        fileWeight = str(WEIGHTTYPE)+str(CULLTHRESH)
    if not os.path.isfile('Results/ParamTests/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv"):
        resFile = open('Results/ParamTests/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
        # resFile.write('sID,R2,RMSE,Pearsons,PearsonsPval,Err,numModels,reproR2,reproRMSE,reproPearsons,reproPearsonsPval,reproErr,numModelsUsed\n')
        resFile.write('sID,Win,Thresh,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,reproErr,reproPearsons,reproPearsonsPval,numModelsUsed,totalLocalModels,totalStableLocalModels\n')
    else:
        resFile = open('Results/ParamTests/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
    resFile.write(str(UID)+','+str(MAX_WINDOW)+','+str(THRESHOLD)+','+
            str(metrics.r2_score(historyData[tLabel],historyData['predictions']))+','+
            str(np.sqrt(metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])))+','+
            str(metrics.mean_absolute_error(historyData[tLabel],historyData['predictions']))+','+
            str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[0])+','+
            str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[1])+','+
            str(len(sourceModels))+','+
            str(metrics.r2_score(p[tLabel],p['predictions']))+','+
            str(np.sqrt(metrics.mean_squared_error(p[tLabel],p['predictions'])))+','+
            str(metrics.mean_absolute_error(p[tLabel],p['predictions']))+',' +
            str(pearsonr(p['predictions'].values,p[tLabel].values)[0])+','+
            str(pearsonr(p['predictions'].values,p[tLabel].values)[1])+','+
            str(meanNumModels)+','+str(len(existingModels))+','+
            str(len(modelMultiHistory.getStableModels(existingModels)))+'\n')

    resFile.close()


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
    global RUNID
    global NUMSTREAMS
    global WEIGHTTYPE
    global UID
    global CULLTHRESH
    global MITHRESH

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
    RUNID = int(sys.argv[13])
    NUMSTREAMS = int(sys.argv[14])
    UID = str(sys.argv[15])
    WEIGHTTYPE = weightType
    CULLTHRESH = float(sys.argv[16])
    MITHRESH = float(sys.argv[17])

    '''
    # set up connection to controller #
    '''

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    s.sendall(ID.encode())
    print("connection established")
    print(ID, PORT, SIM_FROM, SIM_TO, FP, INIT_DAYS, MODEL_HIST_THRESHOLD_ACC, MODEL_HIST_THRESHOLD_PROB, STABLE_SIZE, MAX_WINDOW, THRESHOLD)
    print("model hist acc: "+str(MODEL_HIST_THRESHOLD_ACC))
    print("model hist prob: "+str(MODEL_HIST_THRESHOLD_PROB))
    print("stablesixe: "+str(STABLE_SIZE))
    print(" maxwindow: "+str(MAX_WINDOW))
    print("threshold: "+str(THRESHOLD))

    ack = s.recv(1024).decode()
    print(repr(ack))
    simTarget(weightType,s)

    s.close()


if __name__ == '__main__':main()

