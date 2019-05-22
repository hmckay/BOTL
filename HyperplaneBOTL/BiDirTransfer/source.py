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
from scipy.stats import pearsonr
import os.path


DROP_FIELDS = ['predictions']#['date','year','month','day','predictions','heatingOn']
tLabel='y'

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
sentFlag = 0
UID = ''


def calcError(y,preds):
    y= np.round(y,decimals=1)
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
    # for i in range(0,numPackets):
        # newsection = s.recv(1024)
        # print "length of pickle of seg "+str(i)+": "+str(len(newsection))
        # print newsection
        # pickledModel = pickledModel + newsection
    # print "length of final pickle seg: "+str(len(newsection))
    while (len(pickledModel) < lenofModel):
        pickledModel = pickledModel+(s.recv(1024))
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
    startDate = df.index.min()
    startDate = df.index.min()+INIT_DAYS+1
    startDoW = df.index.min()%20#,'dayOfWeek']

    modelStart = []
    modelStart.append(startDate)
    p = pd.DataFrame(columns = df.columns)#['date','time','oTemp','rain','dTemp','heatingOn','year','month','day','dayOfWeek','predictions'])

    for idx in df.index:
        #print idx
        conceptSize+=1
        if idx%20 == startDoW and storeFlag == False:
            storeFlag = True
            week += 1
        
        if ofUse >= STABLE_SIZE and not sentFlag:
            print("trying to send "+str(modelID))
            sentFlag = modelReadyToSend(modelID,targetModel,s)
        
        #first window of data
        if len(historyData) < MAX_WINDOW:
            if len(historyData)%5 == 4 and len(sourceModels)>0:
                weight = modelMulti.updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,weightType)
                print(weight)
            if not sourceModels:
                #print "prediction is 0.5"
                prediction = modelMulti.defaultInstancePredict(df,idx)
                targetPred = prediction
            else:
                #"making prediction, source Models: "+str(sourceModels)
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
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
               resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
               wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
               if not sourceModels:
                   numModels = 0
               else:
                   modelWeights = weights['sourceR2s']
                   numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
               resultFile.write(str(idx)+','+str(wErr)+','+str(numModels)+'\n')
               resultFile.close()
        #one window of data received
        elif (len(historyData) == MAX_WINDOW) and not buildTMod:
            if weightType != 'OLS' and weightType != 'OLSFE' and weightType != 'OLSFEMI' and weightType != 'Ridge' and weightType != 'NNLS':
                weights['sourceR2s']=weight
                weights['totalR2']=sum(weight.values())
            else: 
                weights = weight
            #weights['sourceR2s']=weight
            #weights['totalR2']=sum(weight.itervalues())
            multiWeights.append(weights)
            #print "building first target model"
            buildTMod = True
            print("building target model: "+str(idx))
            #receive models from controller
            sourceModels = readyToReceive(s,sourceModels)
            targetModel = createModel.createPipeline(historyData,tLabel,DROP_FIELDS)
            #might not want this - wait for it to be used for stable size
            #modelReadyToSend(modelID,targetModel,s)

            weights = modelMulti.calcWeights(historyData,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
               resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
               wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
               modelWeights = weights['sourceR2s']
               numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
               resultFile.write(str(idx)+','+str(wErr)+','+str(numModels+1)+'\n')
               resultFile.close()
            print("first prediction with target model: "+str(idx))
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
            
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            window = p.copy()
            
            #calculate ensemble weights - return dict

            print("first alpha value is: " +str(weights))

            ##########CREATE NEW HISTORY FOR ENSEMBLE WEIGHTS
            existingModels,transitionMatrix,modelID,modelOrder = modelMultiHistory.newHistory(MODEL_HIST_THRESHOLD_ACC,
                    MODEL_HIST_THRESHOLD_PROB,STABLE_SIZE,sourceModels,targetModel,startDate)
            drifts[modelCount] = week
            modelCount = 1
            
        #continue with repro
        else:
            if idx%20 == 0 and (weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'OLSFEMI' or weightType == 'Ridge'):
                weights = modelMulti.calcWeights(historyData[(-1*STABLE_SIZE):],sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
            ofUse += 1
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)
            if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI':
               resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
               #wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
               if len(historyData)<30:
                   wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
               else:
                   wErr = metrics.r2_score(historyData.iloc[(idx-31):][tLabel],historyData.iloc[(idx-31):]['predictions'])
               modelWeights = weights['sourceR2s']
               numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
               resultFile.write(str(idx)+','+str(wErr)+','+str(numModels+1)+'\n')
               resultFile.close()
            prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
            historyData = historyData.append(prediction)
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            p = p.append(targetPred) 
            if idx%50 == 0:
                print(idx, weights)
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
                        #print startDate
                        startDate = idx#window.loc[idx,'date'].date()
                        modelStart.append(startDate)
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
                        modelID, existingModels,transitionMatrix,modelOrder,targetModel = modelMultiHistory.nextModels(existingModels,transitionMatrix, 
                                modelOrder,partial,modelID,tLabel,DROP_FIELDS,startDate,False)
                        # send to controller
                        ofUse = 0
                        sentFlag = 0
                        weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType)
                        #window = modelMulti.initialPredict(window,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
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
    print("number concepts: "+str(modelCount))
    print(modelStart)
    print(modelOrder)
    print(existingModels)
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

    '''
    if not sourceModels:
        #no sourceModels then have to wait to get data
        historyData,futureData,model = initiate(df,tLabel,SIM_FROM,DROP_FIELDS)
        startIDx = futureData.index.min()
        startDate = startIDx#futureData.loc[startIDx,'date'].date()
        existingModels,transitionMatrix,modelID,modelOrder = modelHistory.newHistory(MODEL_HIST_THRESHOLD_ACC,MODEL_HIST_THRESHOLD_PROB,STABLE_SIZE,model,startDate)
        modelReadyToSend(modelID,model,s)
    '''
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
    print("======================")
    print("OVERALL PERFORMANCE")
    print("R2: " + str(metrics.r2_score(historyData[tLabel],historyData['predictions'])))
    print("RMSE: "+str(np.sqrt(metrics.mean_squared_error(historyData[tLabel],historyData['predictions']))))
    print("======================")
    #print historyData
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
    global UID

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
    UID = os.path.splitext(os.path.basename(FP))[0]

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

