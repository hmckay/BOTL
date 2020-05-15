############################################################################
# This is the main script used by each domain. Drifts are detected from 
# this file, models are managed, and transferred to the controller
############################################################################
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
import os.path
from scipy.stats import pearsonr

DROP_FIELDS = ['date','year','month','day','predictions','heatingOn']
tLabel='dTemp'

INIT_DAYS = 0#14
MODEL_HIST_THRESHOLD_ACC = 0#.6
MODEL_HIST_THRESHOLD_PROB = 0#.66
STABLE_SIZE = 0#14
MAX_WINDOW = 0#14
THRESHOLD = 0#.65
PORT = 0
SIM_FROM = 0
SIM_TO = 0
FP = ''
modelsSent = []
sentFlag=0
UID=''


def calcError(y,preds):
    # y= np.round(y,decimals=1)
    #return metrics.mean_squared_error(y,preds)
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
    newsection = ""
    # for i in range(0,numPackets):
        # newsection = s.recv(1024)
        # print "length of pickle of seg "+str(i)+": "+str(len(newsection))
        # print newsection
        # pickledModel = pickledModel + newsection
    # print "length of final pickle seg: "+str(len(newsection))
    while (len(pickledModel) < lenofModel):
        pickledModel = pickledModel+s.recv(1024)
    s.sendall(('ACK,'+str(sourceModID)).encode())

    return storeSourceModel(sourceModID,pickledModel,sourceModels)

def storeSourceModel(sourceModID,pickledModel,sourceModels):
    print("picked model len is: "+str(len(pickledModel)))
    model = pickle.loads(pickledModel)
    sourceModels[sourceModID] = model
    print(model)
    return sourceModels




def runRePro(df,sourceModels,tLabel,weightType,s):
    buildTMod = False
    sourceAlpha = 1
    targetModel = None
    existingModels = dict()
    transitionMatrix = dict()
    multiWeights = []
    numModelsUsed = []
    weight = dict()
    weights = dict()
    for m in sourceModels:
        weight[m] = 1.0/len(sourceModels)
    print("initial weights:")
    print(weight)
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
    window = pd.DataFrame(columns = df.columns)# ['date','time','oTemp','rain','dTemp','heatingOn','year','month','day','dayOfWeek','predictions'])
    historyData = pd.DataFrame(columns = df.columns)# ['date','time','oTemp','rain','dTemp','heatingOn','year','month','day','dayOfWeek','predictions'])
    startDate = df.loc[df.index.min(),'date'].date()
    startDoW = df.loc[df.index.min(),'dayOfWeek']
    initIDX = 0

    modelStart = []
    modelStart.append(startDate)
    p = pd.DataFrame(columns = df.columns)#['date','time','oTemp','rain','dTemp','heatingOn','year','month','day','dayOfWeek','predictions'])


    for idx in df.index:
        conceptSize+=1
        if df.loc[idx,'dayOfWeek'] == startDoW and storeFlag == False:
            storeFlag = True
            week += 1
        # if ofUse >= STABLE_SIZE and not sentFlag:
        if (modelMultiHistory.getLenUsedFor(modelID,existingModels)) and (modelID not in modelsSent):
            print("trying to send "+str(modelID))
            sentFlag = modelReadyToSend(modelID,targetModel,s)
        #first window of data
        if len(historyData['date'].unique()) < MAX_WINDOW:
            # if len(historyData['date'].unique())%7 == 6 and len(sourceModels)>0:
            if len(historyData['date'].unique())>1 and (len(historyData['date'].unique())%(int(MAX_WINDOW/4)) == (int(MAX_WINDOW/4)-1)) and len(sourceModels)>0:
                weight = modelMulti.updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,weightType)
            if not sourceModels:
                prediction = modelMulti.defaultInstancePredict(df,idx)
                targetPred = prediction
            else:
                prediction = modelMulti.initialInstancePredict(df,idx,sourceModels,weight,tLabel,DROP_FIELDS,weightType)
                targetPred = modelMulti.defaultInstancePredict(df,idx)
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
                if not sourceModels:
                    numModels = 0
                else:
                    modelWeights = weights['sourceR2s']
                    numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                    numModelsUsed.append(numModels)
            #p = p.append(modelMulti.defaultInstancePredict(df,idx))
            #print week, df.loc[idx,'dayOfWeek']
            #window = window.append(prediction)
            #windowSize = len(window['date'].unique())
        #one window of data received
        elif (len(historyData['date'].unique()) == MAX_WINDOW) and not buildTMod:
            if weightType != 'OLS' and weightType != 'OLSFE' and weightType != 'OLSFEMI':
                weights['sourceR2s']=weight
                weights['totalR2']=sum(weight.values())
            else: 
                weights = weight
            #weights['sourceR2s']=weight
            #weights['totalR2']=sum(weight.itervalues())
            multiWeights.append(weights)
            #print historyData
            #print "building first target model"
            buildTMod = True
            sourceModels = readyToReceive(s,sourceModels)
            targetModel = createModel.createPipeline(historyData,tLabel,DROP_FIELDS)

            weights = modelMulti.calcWeights(historyData,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            
            #print sourceModels
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
            
            historyData = historyData.append(prediction)
            # print(historyData)
            p = p.append(targetPred)
            window = p.copy()
            initIDX = idx
            #calculate ensemble weights - return dict
            #weights = modelMulti.calcWeights(historyData,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)

            print("first alpha value is: " +str(weights))

            ##########CREATE NEW HISTORY FOR ENSEMBLE WEIGHTS
            existingModels,transitionMatrix,modelID,modelOrder = modelMultiHistory.newHistory(MODEL_HIST_THRESHOLD_ACC,
                    MODEL_HIST_THRESHOLD_PROB,STABLE_SIZE,sourceModels,targetModel,startDate)
            drifts[modelCount] = week
            modelCount = 1
            
        #continue with repro
        else:
            # if len(historyData['date'].unique())%14 == 0 and (weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'OLSFEMI'):
            if (len(historyData['date'].unique())%(int(MAX_WINDOW/2)) == 0 and (len(window)>0) and 
                    (df.loc[idx,'date']!=df.loc[idx-1,'date']) and ('OLS' in weightType or weightType == 'Ridge')):
                weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
            #if (not (df.loc[idx,'date'] == df.loc[idx-1,'date'])) and (len(window)>0):
                #RECALC ENSEMBLE WEIGHT?
                ########sourceAlpha = gotl.updateAlpha(window,sourceModel,targetModel,sourceAlpha,tLabel,DROP_FIELDS)
            ############### ENSEMBLE WEIGHTS PREDICT
            #weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
            ofUse+=1
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
            historyData = historyData.append(prediction)
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            p = p.append(targetPred)
        
            diff = abs(targetPred['predictions']-targetPred[tLabel])

            if diff >0.5 and windowSize ==0:
                window = window.append(targetPred)
                windowSize = len(window['date'].unique())
            elif windowSize>0:
                window = window.append(targetPred)
                windowSize = len(window['date'].unique())
                if windowSize == MAX_WINDOW:
                    err = calcError(window[tLabel],window['predictions'])
                    if err < THRESHOLD:
                        lastModID = modelID
                        modelCount+=1
                        #print startDate
                        startDate = window.loc[idx,'date'].date()
                        modelStart.append(startDate)
                        conceptSize = len(window)
                    
                        partial = window.copy()
                        #CALC ENSEMBLE WEIGHT
                        sourceModels=readyToReceive(s,sourceModels)
                        modelID, existingModels,transitionMatrix,modelOrder,targetModel = modelMultiHistory.nextModels(existingModels,transitionMatrix, 
                                modelOrder,partial,modelID,tLabel,DROP_FIELDS,startDate,False)
                        weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
                        ofUse = 0
                        sentFlag = 0
                        #print weights
                        ############sourceAlpha = gotl.updateAlpha(partial,sourceModel,targetModel,sourceAlpha,tLabel,DROP_FIELDS)
                        #modelID,existingModels,transitionMatrix,modelOrder,model = modelHistory.nextModels(existingModels,transitionMatrix,modelOrder,partial,
                        #        modelID,tLabel,DROP_FIELDS,startDate,False)
                        #modelID,existingModels,transitionMatrix,modelOrder,targetModel,sourceAlpha = gotlHistory.nextModels(existingModels,transitionMatrix,modelOrder,partial,
                        #        modelID,sourceAlpha,tLabel,DROP_FIELDS,startDate,False)
                        #window = modelMulti.initialPredict(window,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
                        window = createModel.initialPredict(window,targetModel,tLabel,DROP_FIELDS)
                        #window = gotl.initialPredict(window,sourceModel,targetModel,sourceAlpha,tLabel,DROP_FIELDS)
                        #print startDate, sourceAlpha
                        if lastModID < modelID:
                            drifts[modelCount] = week
                        #print 'finished here'
                        #model = createModel.createPipeline(window,tLabel,DROP_FIELDS)
                        #window = createModel.initialPredict(window,model,tLabel,DROP_FIELDS)
            if windowSize>=MAX_WINDOW:
                top = window.loc[window.index.min()]
                diffTop = abs(top['predictions']-top[tLabel])

                while (windowSize>=MAX_WINDOW or diffTop <=0.5):
                    #print window
                    #print "stuck"
                    window = window.drop([window.index.min()],axis=0)
                    windowSize = len(window['date'].unique())
                    if windowSize == 0:
                        break
                    top = window.loc[window.index.min()]
                    diffTop = abs(top['predictions']-top[tLabel])

        if df.loc[idx,'dayOfWeek'] == (startDoW-1) and storeFlag == True:
            storeFlag = False
            #print "LOOK BELOW"
            #print week, sourceAlpha
            multiWeights.append(weights['sourceR2s'])
            #if week == 3:
            #    break
    endDate = historyData.loc[historyData.index.max(),'date'].date()
    modelOrder,existingModels = modelMultiHistory.updateLastModelUsage(modelOrder,existingModels,endDate)
    print("number concepts: "+str(modelCount))
    print(modelStart)
    print(modelOrder)
    print(existingModels)
    print(transitionMatrix)
    mse = metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])
    print(mse)
    print(calcError(historyData[tLabel],historyData['predictions']))
    print(multiWeights)
    #print drifts
    #print alphas
    #makeGraph.plotAlpha(alphas,drifts)
    print(metrics.r2_score(historyData[tLabel],historyData['predictions']))
    #partialp = p.loc[initIDX:,:]
    #print partialp
    print(metrics.r2_score(p[tLabel],p['predictions']))

    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed





def simTarget(weightType,s):
    sourceModels = dict()
    sourceModels = readyToReceive(s,sourceModels)
    #get models from controller
    #sourceModels,sourceTM = source.simSource(source_From,source_To,source_FP,dinit,threshAcc,threshProb,stable,mwind,thresh)
    print("source Models:")
    print(sourceModels)
    df = preprocess.pullData(FP)
    df = preprocess.subsetData(df,SIM_FROM,SIM_TO)
    df['predictions'] = np.nan
    
    historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed = runRePro(df,sourceModels,tLabel,weightType,s)
    numModelsUsedArr = np.array(numModelsUsed)
    meanNumModels = numModelsUsedArr.mean()
    # print "======================"
    # print "OVERALL PERFORMANCE"
    # print "R2: " + str(metrics.r2_score(historyData[tLabel],historyData['predictions']))
    # print "RMSE: "+str(np.sqrt(metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])))
    # print "======================"
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

    # if not os.path.isfile('KDDResults/'+str(weightType)+'/resultsStreams.csv'):
        # resFile = open('KDDResults/'+str(weightType)+'/resultsStreams.csv','a')
        # resFile.write('sID,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,reproErr,reproPearsons,reproPearsonsPval,numModelsUsed\n')
    # else:
        # resFile = open('KDDResults/'+str(weightType)+'/resultsStreams.csv','a')
    # # if not os.path.isfile('KDDResults/'+str(WEIGHTTYPE)+'/results'+str(NUMSTREAMS)+"streams.csv"):
        # # resFile = open('KDDResults/'+str(WEIGHTTYPE)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
        # # resFile.write('R2,RMSE,Err,numModels,reproR2,reproRMSE,reproErr,sID\n')
    # # else:
        # # resFile = open('KDDResults/'+str(WEIGHTTYPE)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
    # resFile.write(str(UID)+','+str(metrics.r2_score(historyData[tLabel],historyData['predictions']))+','+
            # str(np.sqrt(metrics.mean_squared_error(historyData[tLabel],historyData['predictions'])))+','+
            # str(metrics.mean_absolute_error(historyData[tLabel],historyData['predictions']))+','+
            # str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[0])+','+
            # str(pearsonr(historyData['predictions'].values,historyData[tLabel].values)[1])+','+
            # str(len(sourceModels))+','+
            # str(metrics.r2_score(p[tLabel],p['predictions']))+','+
            # str(np.sqrt(metrics.mean_squared_error(p[tLabel],p['predictions'])))+','+
            # str(metrics.mean_absolute_error(p[tLabel],p['predictions']))+',' +
            # str(pearsonr(p['predictions'].values,p[tLabel].values)[0])+','+
            # str(pearsonr(p['predictions'].values,p[tLabel].values)[1])+','+
            # str(meanNumModels)+'\n')

    # resFile.close()
    
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
    SIM_FROM = datetime.strptime(sys.argv[3],"%Y-%m-%d").date()
    SIM_TO = datetime.strptime(sys.argv[4],"%Y-%m-%d").date()
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
    # global ID
    # global PORT
    # global SIM_FROM
    # global SIM_TO
    # global FP
    # global INIT_DAYS
    # global MODEL_HIST_THRESHOLD_ACC
    # global MODEL_HIST_THRESHOLD_PROB
    # global STABLE_SIZE
    # global MAX_WINDOW
    # global THRESHOLD
    # global UID

    # HOST = 'localhost'
    # ID = str(sys.argv[1])
    # PORT = int(sys.argv[2])
    # SIM_FROM = datetime.strptime(sys.argv[3],"%Y-%m-%d").date()
    # SIM_TO = datetime.strptime(sys.argv[4],"%Y-%m-%d").date()
    # FP = str(sys.argv[5])
    # INIT_DAYS = int(sys.argv[6])
    # MODEL_HIST_THRESHOLD_ACC = float(sys.argv[7])
    # MODEL_HIST_THRESHOLD_PROB = float(sys.argv[8])
    # STABLE_SIZE = int(sys.argv[9])
    # MAX_WINDOW = int(sys.argv[10])
    # THRESHOLD = float(sys.argv[11])
    # weightType = str(sys.argv[12])
    # UID = str(SIM_FROM)+'-'+str(SIM_TO)

    '''
    # set up connection to controller #
    '''

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    s.sendall(ID.encode())
    print("connection established")
    print(ID, PORT, SIM_FROM, SIM_TO, FP, INIT_DAYS, MODEL_HIST_THRESHOLD_ACC, MODEL_HIST_THRESHOLD_PROB, STABLE_SIZE, MAX_WINDOW, THRESHOLD)

    ack = s.recv(1024).decode()
    print(repr(ack))
    simTarget(weightType,s)

    s.close()







'''
def main():
    cdDetection = True
    source_From = datetime.strptime('2014-04-01',"%Y-%m-%d").date()
    source_To =datetime.strptime('2014-09-20',"%Y-%m-%d").date()
    target_From = datetime.strptime('2015-01-01',"%Y-%m-%d").date()
    target_To =datetime.strptime('2015-09-15',"%Y-%m-%d").date()
    target_FP = '../Datasets/userTARGETDataSimulation.csv'
    source_FP =  '../Datasets/userSOURCEDataSimulation.csv'
    sourceModels,sourceTM = source.simSource(source_From,source_To,source_FP)
    df = preprocess.pullData(target_FP)
    df = preprocess.subsetData(df,target_From,target_To)
    df['predictions'] = np.nan
    
    runRePro(df,sourceModels,tLabel)
'''
if __name__ == '__main__':main()

