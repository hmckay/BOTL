
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
from Models import modelMultiConceptTransferHistoryAWPro as modelMultiHistory
import preprocessData as preprocess
from datetime import datetime,timedelta
from sklearn import metrics
from scipy.stats import pearsonr
from pyadwin import Adwin
from utilityFunctions2 import *
# from utilityFunctions import *



DROP_FIELDS = []#['Time','predictions']
tLabel=''#[]#'FollowingTime'

INIT_DAYS = 0
STABLE_SIZE = 0
MAX_WINDOW = 0
ADWIN_DELTA = 0
MODEL_HIST_THRESHOLD_ACC = 0
MODEL_HIST_THRESHOLD_PROB = 0
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
DEFAULT_PRED = ''
EPSILON = 0
PATHRESH = 0
PCAVAR = 0
METASTATS = dict()
COMPSTATS = dict()
COMMSTATS = dict()
LOCALMODELSTATS = dict()
TRANSMODELSTATS = dict()
REPLACEMENTMODELSTATS = dict()
OVERVIEWSTATS= dict()
MODELUSAGE=dict()
LEARNER_TYPE = ''
FRAMEWORKTYPE = ''
CONN_DEG = 0
NEIGHBORS = ""
REPLACEMENT=0
REPLACEMENTSTR=""


# def calcError(y,preds):
    # return metrics.r2_score(y,preds)

# def modelReadyToSend(modelID,model,s):
    # successFlag = 0
    # if modelID not in modelsSent:
        # successFlag = handshake(modelID,model,s)
    # else:
        # print("model already sent")
        # return 1

    # if successFlag:
        # modelsSent.append(modelID)
        # print("sucessfully sent model")
        # return 1
    # else:
        # print("unsucessful send")
        # return 0

# def handshake(modelID,model,s):
    # print(modelID)
    # modelToSend = pickle.dumps(model)
    # lenofModel = len(modelToSend)
    # brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
    # numPackets = len(brokenBytes)
    # RTSmsg = 'RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel)
    # s.sendall(RTSmsg.encode())
    # ack = s.recv(1024).decode()
    # ackNumPackets = int(ack.split(',')[1])

    # if ackNumPackets == numPackets:
        # return sendModel(modelID,brokenBytes,s)
    # return 0

# def sendModel(modelID,brokenBytes,s):
    # for i in brokenBytes:
        # s.sendall(i)
        # print("finished sending")
    # recACK = s.recv(1024).decode()
    # if modelID == int(recACK.split(',')[1]):
        # return 1
    # return 0

# def readyToReceive(s,sourceModels):
    # s.sendall(('RTR,'+str(ID)).encode())
    # data = s.recv(1024).decode()
    # print("TARGET: num models to receive "+repr(data))
    # ACKFlag = data.split(',')[0]
    # numModels = int(data.split(',')[1])
    # print("ready to recieve function called")
    # if ACKFlag == 'ACK':
        # if numModels == 0:
            # s.sendall(('END').encode())
            # return sourceModels
        # s.sendall(('ACK').encode())
        # for i in range(0,numModels):
            # sourceModels = receiveModels(s,sourceModels)
    # return sourceModels

# def receiveModels(s,sourceModels):
    # RTSInfo = s.recv(1024).decode()
    # RTSFlag = RTSInfo.split(',')[0]
    # sourceModID = RTSInfo.split(',')[1]
    # numPackets = int(RTSInfo.split(',')[2])
    # lenofModel = int(RTSInfo.split(',')[3])
    # print("NUMBER OF PACKETS EXPECTING: "+str(numPackets))
    # s.sendall(('ACK,'+str(numPackets)).encode())
    
    # pickledModel = b''
    # while (len(pickledModel) < lenofModel):
        # pickledModel = pickledModel+s.recv(1024)
    # s.sendall(('ACK,'+str(sourceModID)).encode())

    # return storeSourceModel(sourceModID,pickledModel,sourceModels)

# def storeSourceModel(sourceModID,pickledModel,sourceModels):
    # print("picked model len is: "+str(len(pickledModel)))
    # model = pickle.loads(pickledModel)
    # sourceModels[sourceModID] = model
    # print(model['model'])
    # return sourceModels


def runAWPro(df,sourceModels,tLabel,weightType,s):
    global modelsSent
    global METASTATS
    global MODELUSAGE
    adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
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
    conceptSimPreds = pd.DataFrame()
    conceptuallySimGroup = []
    startIDX = df.index.min()+INIT_DAYS+1
    # startDoW = df.index.min()%20
    startDoW = df.index.min()%INIT_DAYS
    retrainFlag=False
    updateFlag=False
    tempModel = False
    distanceMatrix = None
    localDistanceMatrix = None
    affinityMatrix = None
    existingBases = None
    newTarget = False
    groupedNames = None
    modelSet = dict()
    orderedModels = []
    notSent = []
    delaySend = []
    numBases=0
    currentBases=[]
    METASTATS['currentBaseLearnerIDs'] = []
    delayedTransferList = []
    recluster = False

    modelStart = []
    modelStart.append(startIDX)
    p = pd.DataFrame(columns = df.columns)
    historyData = historyData.astype(df.dtypes)
    window = window.astype(df.dtypes)
    
    if 'AWE' in weightType:
        for k in sourceModels.keys():
            modelSet[k] = sourceModels[k]
            modelSet[k]['delAWE'] = 0
        # modelSet = AWEaddToModelSet(df,None,None,sourceModels,modelSet,tLabel,DROP_FIELDS)
    elif 'AddExp' in weightType:
        print("here initinti")
        # modelSet,orderedModels,weights['metaModel'] = AddExpOaddToModelSet(df,0,None,None,sourceModels,modelSet,
                # orderedModels,weights,tLabel,DROP_FIELDS)
    else:
        # modelSet = addToModelSet(None,None,sourceModels,modelSet)
        initModelSet = modelSet.copy().keys()
        modelSet,sourceModels,distanceMatrix,tc,METASTATS = addToModelSet(None,None,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,STABLE_SIZE,distanceMatrix,REPLACEMENT)
        if not set(initModelSet) == set(modelSet.keys()):
            recluster = True
        METASTATS['totalPACalcs']+=tc
        if tc > 0:
            METASTATS['triggerPACalcs']+=1

    for idx in df.index:
        METASTATS['totalNumInstances']+=1
        if modelID not in MODELUSAGE.keys():
            MODELUSAGE[modelID]=0
        if MODELUSAGE[modelID]>=STABLE_SIZE and modelID not in METASTATS['stableLocal']:
            METASTATS['stableLocal'].append(modelID)
        conceptSize+=1
        if idx%INIT_DAYS == startDoW and storeFlag == False:
            storeFlag = True
            week += 1
        
        # if ofUse >= STABLE_SIZE and not sentFlag:
        # if (modelMultiHistory.getLenUsedFor(modelID,existingModels)+ofUse>=STABLE_SIZE) and (modelID not in modelsSent):
        # if (modelMultiHistory.getLenUsedFor(modelID,existingModels)>=STABLE_SIZE) and (modelID not in modelsSent):
        # if (ofUse >=STABLE_SIZE) and (modelID not in modelsSent) and (modelID not in notSent):
        # pastUse = 0
        # if modelID in existingModels.keys():
            # pastUse = existingModels[modelID]['ofUse']
        if (ofUse) >= STABLE_SIZE and (uniqueModID(modelID,UID) not in modelsSent) and (uniqueModID(modelID,UID) not in notSent) and len(delayedTransferList)==0:
            print("trying to send "+str(modelID))
            # if 'OLSKPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData[(-1*MAX_WINDOW):],DROP_FIELDS,True)
            # elif 'OLSPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData[(-1*MAX_WINDOW):],DROP_FIELDS,False)
            # else:
                # PCs = None
            # existingModels[modelID]['PCs'] = PCs
            modelInfoToSend = {'model':targetModel,'PCs':existingModels[modelID]['PCs']}
            # sentFlag,modelsSent = modelReadyToSend(modelID,modelInfoToSend,s,modelsSent)
            METASTATS['LOCALMODELSTATS']['stableModelsLearnt']+=1
            METASTATS['LOCALMODELSTATS']['stableModelIDs'].append(uniqueModID(modelID,UID))
            checkSendFlag = True
            if 'OLSFEMIRed' in weightType:
                checkSendFlag,tc,METASTATS = checkToSendMI(modelID,targetModel,historyData[(-1*MAX_WINDOW):],existingModels,modelsSent,weightType,DROP_FIELDS,METASTATS,tLabel,UID)
            elif 'Red' in weightType:
                checkSendFlag, localDistanceMatrix,tc,METASTATS = checkToSend(modelID,window,existingModels,modelsSent,localDistanceMatrix,
                        weightType,DROP_FIELDS,METASTATS,tLabel,UID)
                METASTATS['totalPACalcs']+=tc
                if tc > 0:
                    METASTATS['triggerPACalcs']+=1
            if not checkSendFlag:
                # notSent.append(modelID)
                notSent.append(uniqueModID(modelID,UID))
                METASTATS['LOCALMODELSTATS']['stableNeverSent']+=1
                METASTATS['LOCALMODELSTATS']['stableNeverSentIDs'].append(uniqueModID(modelID,UID))
            else:
                sentFlag,modelsSent,delayedTransferList,METASTATS = checkSimilarModelTransfer(UID,modelID,modelInfoToSend, conceptuallySimGroup,s,modelsSent,METASTATS)
                # sentFlag,modelsSent = modelReadyToSend(modelID,modelInfoToSend,s,modelsSent)
            if sentFlag and uniqueModID(modelID,UID) not in METASTATS['modelsSent']:
                METASTATS['modelsSent'].append(uniqueModID(modelID,UID))
            # if sentFlag and modelID not in METASTATS['modelsSent']:
                # METASTATS['modelsSent'].append(modelID)
        
        #first window of data
        if len(historyData) < MAX_WINDOW:
            if (len(historyData)%(int(MAX_WINDOW/4)) == (int(MAX_WINDOW/4)-1)) and len(sourceModels)>0:
                weight,METASTATS = modelMulti.updateInitialWeights(historyData,sourceModels,tLabel,
                        DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH)
                if 'PAC' in weightType or 'CL2' in weightType:
                    METASTATS['totalPACalcs']+=weight[-1][1]
                    if weight[-1][1] > 0:
                        METASTATS['triggerPACalcs']+=1
                    METASTATS['totalOtherMetricCalcs']+=weight[-1][0]
                    if weight[-1][0]>0:
                        METASTATS['triggerMetricCalcs']+=1
                else:
                    METASTATS['totalOtherMetricCalcs']+=weight[-1]
                    if weight[-1]>0:
                        METASTATS['triggerMetricCalcs']+=1
                METASTATS['metaWeightUpdate']+=1
            if not sourceModels:
                prediction,numBases,currentBases = modelMulti.defaultInstancePredict(df,idx,DEFAULT_PRED)
                METASTATS['numberBaseLearners'].append(numBases)
                METASTATS['baseLearnerList'].append(currentBases)
                if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                    METASTATS['baseLearnerChange']+=1
                    METASTATS['currentBaseLearnerIDs']=currentBases
                targetPred = prediction
            else:
                if ('PAC' in weightType or 'CL' in weightType) and len(sourceModels)>=1 and len(historyData)>=20:
                    prediction,distanceMatrix,affinityMatrix,groupedNames,numBases,currentBases,tc,METASTATS = modelMulti.initialInstancePredict(df,idx,
                            sourceModels,weight,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,distanceMatrix,
                            affinityMatrix,groupedNames)
                    METASTATS['numberBaseLearners'].append(numBases)
                    METASTATS['baseLearnerList'].append(currentBases)
                    if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                        METASTATS['baseLearnerChange']+=1
                        METASTATS['currentBaseLearnerIDs']=currentBases
                else:
                    print("in this else because not")
                    prediction,numBases,currentBases,tc,METASTATS = modelMulti.initialInstancePredict(df,idx,sourceModels,weight,tLabel,
                            DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,distanceMatrix,affinityMatrix,groupedNames)
                    METASTATS['numberBaseLearners'].append(numBases)
                    METASTATS['baseLearnerList'].append(currentBases)
                    if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                        METASTATS['baseLearnerChange']+=1
                        METASTATS['currentBaseLearnerIDs']=currentBases
                #"making prediction, source Models: "+str(sourceModels)
                # prediction = modelMulti.initialInstancePredict(df,idx,sourceModels,weight,tLabel,
                        # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH)
                targetPred,numBases,currentBases = modelMulti.defaultInstancePredict(df,idx,DEFAULT_PRED)
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
            # sourceModels = readyToReceive(s,sourceModels,ID)
            sourceModels,METASTATS = readyToReceive(s,sourceModels,UID,METASTATS)
            if len(modelSet.keys())>0:
                initModelSet = modelSet.copy().keys()
                existingModels,sourceModels,modelSet,distanceMatrix = removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix)
                if not set(initModelSet) == set(modelSet.keys()):
                    recluster = True
            targetModel,METASTATS = createModel.createPipeline(historyData,tLabel,DROP_FIELDS,DEFAULT_PRED,LEARNER_TYPE,METASTATS)
            if 'AWE' in weightType:
                modelSet,METASTATS = AWEaddToModelSet(historyData,None,None,sourceModels,modelSet,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
            elif 'AddExpO' in weightType:
                # if 'metaModel' not in weights.keys():
                weights['metaModel']=dict()
                modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpOaddToModelSet(historyData,idx,None,None,sourceModels,modelSet,
                        orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                METASTATS['metaWeightUpdate']+=1
                # weights['metaModel'][modelID]=1
                # weights = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,modelID,weights)
            elif 'AddExpP' in weightType:
                # if 'metaModel' not in weights.keys():
                weights['metaModel']=dict()
                modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpPaddToModelSet(historyData,idx,None,None,sourceModels,modelSet,
                        orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                METASTATS['metaWeightUpdate']+=1

            else:
                # modelSet = addToModelSet(None,None,sourceModels,modelSet)
                modelSet,sourceModels,distanceMatrix,tc,METASTATS = addToModelSet(None,None,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,STABLE_SIZE,distanceMatrix,REPLACEMENT)
            # if 'OLSKPAC' in weightType:
                # PCs = modelMultiHistory.getPCs(historyData,DROP_FIELDS,PCAVAR)
            # elif 'OLSPAC' in weightType:
            newTarget = True
            recluster = True
            if 'PA' in weightType:
                PCs = modelMultiHistory.getPCs(historyData,DROP_FIELDS,PCAVAR)
            else:
                PCs = None

            if 'AddExp' not in weightType:
                currentBases = []
                if type(weights)is dict and 'metaXColumns' in weights.keys():
                    currentBases = weights['metaXColumns'].tolist()
                # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(historyData,modelSet,
                        # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,None,modelID,PCs,
                        # distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                weights,distanceMatrix,affinityMatrix,groupedNames,tc,METASTATS = modelMulti.calcWeights(historyData,modelSet,
                        targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,None,modelID,PCs,
                        distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,recluster)
                print("recalcWeights4")
                if 'PAC' in weightType or 'CL2' in weightType:
                    METASTATS['totalPACalcs']+=tc[1]
                    if tc[1] > 0:
                        METASTATS['triggerPACalcs']+=1
                    METASTATS['totalOtherMetricCalcs']+=tc[0]
                    if tc[0]>0:
                        METASTATS['triggerMetricCalcs']+=1
                else:
                    METASTATS['totalOtherMetricCalcs']+=tc
                    if tc>0:
                        METASTATS['triggerMetricCalcs']+=1
                METASTATS['metaWeightUpdate']+=1
            # if 'AddExp' in weightType:
                # weights['metaModel'] = dict()
                # for mID in modelSet.keys():
                    # weights['metaModel'][mID]=1
                # if modelID not in weights['metaModel'].keys():
                    # weights = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,modelID,weights)
                # else:
                    # weights = modelMulti.calcUpdateAddExpWeight(df,idx,modelSet,weights['metaModel'],targetModel,
                            # modelID,tLabel,DROP_FIELDS)
            if 'AWE' in weightType:
                modelSet = weights['AWEmodelSet']
            newTarget = False
            recluster = False
            # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            if 'OLS' in weightType and 'AddExp' not in weightType:
                modelWeights = weights['sourceR2s']
                numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                numModelsUsed.append(numModels)

            # weights = modelMulti.calcWeights(historyData,sourceModels,targetModel,tLabel,
                    # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,None,None,PCs)
            # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(historyData,sourceModels,
                    # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,None,None,PCs,
                    # None,None,newTarget,groupedNames)
            # newTarget = False
            # # if weightType == 'OLSFE' or weightType =='OLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
            # if 'OLS' in weightType:
                # modelWeights = weights['sourceR2s']
                # numModels = sum(1 for val in list(modelWeights.values()) if val != 0)
                # numModelsUsed.append(numModels)
            # # if weightType == 'OLSFE' or weightType =='OLS':
            # #    # resultFile = open('performanceVsCost3'+str(ID)+str(weightType)+'.csv','a')
            # #    # wErr = metrics.r2_score(historyData[tLabel],historyData['predictions'])
            # #    # modelWeights = weights['sourceR2s']
            # #    # numModels = sum(1 for val in modelWeights.values() if val != 0)
            # #    # resultFile.write(str(idx)+','+str(wErr)+','+str(numModels+1)+'\n')
            # #    # resultFile.close()
            print("first prediction with target model: "+str(idx))
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            prediction,numBases,currentBases = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType,None,modelID,PCs)
            METASTATS['numberBaseLearners'].append(numBases)
            METASTATS['baseLearnerList'].append(currentBases)
            if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                METASTATS['baseLearnerChange']+=1
                METASTATS['currentBaseLearnerIDs']=currentBases
            ofUse+=1
            
            historyData = historyData.append(prediction)
            p = p.append(targetPred)
            
            #calculate ensemble weights - return dict

            print("first alpha value is: " +str(weights))

            ##########CREATE NEW HISTORY FOR ENSEMBLE WEIGHTS
            existingModels,transitionMatrix,modelID,modelOrder = modelMultiHistory.newHistory(MODEL_HIST_THRESHOLD_ACC,MODEL_HIST_THRESHOLD_PROB,
                    STABLE_SIZE,sourceModels,targetModel,startIDX,PCs)
            if modelID not in MODELUSAGE.keys():
                MODELUSAGE[modelID]=1
            else:
                MODELUSAGE[modelID]+=1
            drifts[modelCount] = week
            modelCount = 1
            
        #continue with repro
        else:
            if idx%50 ==0: 
                print("using metaX:"+str(weights['metaXColumns']))
                print("modelID: "+str(modelID))
                print("tempModel?: "+str(tempModel))
            if weightType == 'OLSCL2' or 'OLSPAC' in weightType or 'OLSKPAC' in weightType or REPLACEMENT == True:
                stableLocals = modelMultiHistory.getStableModels(existingModels)
            else:
                stableLocals = None
            if 'PA' in weightType and tempModel==False:# or 'OLSKPAC' in weightType:
                PCs = existingModels[modelID]['PCs']
            else:
                PCs = None
            if 'AddExp' in weightType:
                tc =0
                if modelID not in weights['metaModel'].keys():
                    weights,tc = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,modelID,targetModel,weights)
                else:
                    weights,tc = modelMulti.calcUpdateAddExpWeight(df,idx,modelSet,weights['metaModel'],targetModel,
                            modelID,tLabel,DROP_FIELDS)
                METASTATS['totalOtherMetricCalcs']+=tc
                if tc>0:
                    METASTATS['triggerMetricCalcs']+=1
                METASTATS['metaWeightUpdate']+=1
            # if idx%25 == 0 and (weightType == 'OLS' or weightType == 'OLSFE' or 
                    # weightType == 'OLSFEMI' or weightType == 'Ridge' or weightType == 'OLSCL'):
            if idx%(int(MAX_WINDOW/2)) == 0 and ('OLS' in weightType or weightType == 'Ridge'):
                # stableLocals = modelMultiHistory.getStableModels(existingModels)
                # weights = modelMulti.calcWeights(historyData[(-1*STABLE_SIZE):],sourceModels,
                        # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,PCs)
                if 'AddExp' not in weightType:
                    currentBases = []
                    if 'metaXColumns' in weights.keys():
                        currentBases = weights['metaXColumns'].tolist()
                        print("current bases are: "+str(currentBases))
                        print("recluster is:" +str(recluster))
                    weights,distanceMatrix,affinityMatrix,groupedNames,tc,METASTATS = modelMulti.calcWeights(
                            historyData[(-1*STABLE_SIZE):],modelSet,targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,
                            CULLTHRESH,MITHRESH,stableLocals,modelID,PCs,
                            distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,recluster)
                    print("recalcWeights5")
                    print("after bases are:" +str(weights['metaXColumns'].tolist()))
                    print("using model: "+str(modelID))
                    recluster=False
                    # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(
                            # historyData[(-1*STABLE_SIZE):],modelSet,targetModel,tLabel,DROP_FIELDS,weightType,
                            # CULLTHRESH,MITHRESH,stableLocals,modelID,PCs,
                            # distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                    if 'PAC' in weightType or 'CL2' in weightType:
                        METASTATS['totalPACalcs']+=tc[1]
                        if tc[1] > 0:
                            METASTATS['triggerPACalcs']+=1
                        METASTATS['totalOtherMetricCalcs']+=tc[0]
                        if tc[0]>0:
                            METASTATS['triggerMetricCalcs']+=1
                    else:
                        METASTATS['totalOtherMetricCalcs']+=tc
                        if tc>0:
                            METASTATS['triggerMetricCalcs']+=1
                    METASTATS['metaWeightUpdate']+=1
                if 'AWE' in weightType:
                    modelSet = weights['AWEmodelSet']
                # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(historyData[(-1*STABLE_SIZE):],
                        # sourceModels,targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,PCs,
                        # distanceMatrix,affinityMatrix,newTarget,groupedNames)
                newTarget = False
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
            # prediction = modelMulti.instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,
                    # weightType,stableLocals,modelID,PCs)
            prediction,numBases,currentBases = modelMulti.instancePredict(df,idx,modelSet,targetModel,weights,tLabel,DROP_FIELDS,
                    weightType,stableLocals,modelID,PCs)
            MODELUSAGE[modelID]+=1
            METASTATS['numberBaseLearners'].append(numBases)
            METASTATS['baseLearnerList'].append(currentBases)
            if not set(METASTATS['currentBaseLearnerIDs']) == set(currentBases):
                METASTATS['baseLearnerChange']+=1
                METASTATS['currentBaseLearnerIDs']=currentBases
            historyData = historyData.append(prediction)
            targetPred = createModel.instancePredict(df,idx,targetModel,tLabel,DROP_FIELDS)
            p = p.append(targetPred) 
            # if groupedNames and not conceptuallySimGroup:
            # if REPLACEMENT and groupedNames and not conceptuallySimGroup and not tempModel:
            if REPLACEMENT and not conceptuallySimGroup and 'OLSFEMI' in weightType and len(modelSet)>1:
                conceptuallySimGroup,totalCalcs,METASTATS = getMISimModels(modelID,targetModel,historyData[(-1*MAX_WINDOW):],modelSet,METASTATS,DROP_FIELDS,tLabel,stableLocals)
                if modelID not in conceptuallySimGroup:
                    conceptuallySimGroup.append(modelID)
                cols = conceptuallySimGroup.copy()
                cols.append(tLabel)
                conceptSimPreds= pd.DataFrame(columns=cols)
                print(conceptuallySimGroup)
            if REPLACEMENT and groupedNames and len(groupedNames)>1  and not conceptuallySimGroup and not tempModel:
                for g in groupedNames:
                    if modelID in g:
                        conceptuallySimGroup = g
                        cols = conceptuallySimGroup.copy()
                        cols.append(tLabel)
                        conceptSimPreds= pd.DataFrame(columns=cols)
                        print("starting to collect for eval at: "+str(idx))
                        print("modelOrder-1: "+str(modelOrder[-1]))
                        print("modelID: "+str(modelID))
                        break
            # if len(conceptuallySimGroup)>1:
            if REPLACEMENT and len(conceptuallySimGroup)>1 and not tempModel:
                conPreds = dict()
                conPreds[tLabel]=prediction[tLabel].copy()
                conPreds[modelID]=targetPred['predictions'].copy()
                # print("keys in modelset")
                # print(modelSet.keys())
                # print("conceptSimGroup")
                # print(conceptuallySimGroup)
                for m in conceptuallySimGroup:
                    print(m)
                    if m != modelID:
                        if m in modelSet.keys():
                            conPreds[m]=createModel.singleInstancePredict(df.loc[idx],modelSet[m]['model'],tLabel,DROP_FIELDS)
                        else:
                            conPreds[m]=createModel.singleInstancePredict(df.loc[idx],stableLocals[m]['model'],tLabel,DROP_FIELDS)
                        # conPreds[m]=createModel.singleInstancePredict(df.loc[idx],modelSet[m]['model'],tLabel,DROP_FIELDS)
                        # conPreds[m] = createModel.instancePredict(df.copy(),idx,modelSet[m]['model'],tLabel,DROP_FIELDS)['predictions']
                conceptSimPreds = conceptSimPreds.append(conPreds,ignore_index=True)

            if idx%50 == 0:
                print(idx, weights)
            diff = abs(targetPred['predictions']-targetPred[tLabel])

            if retrainFlag==False:
                updateFlag,n1,n0,u1,u0 = adwin.update(diff)
            else:
                updateFlag = False
                window = window.append(prediction)
                if len(window) >= MAX_WINDOW:
                    print("retain Flag is True, and updateFlag is now false")
                    print("finally have enough data to make a new model")
                    print("previous model was: "+str(modelID))
                    lastModID = modelID
                    modelCount+=1
                    startIDX = idx
                    modelStart.append(startIDX)
                    conceptSize = len(window)
                    partial = window.copy()
                    #retrain model
                    # sourceModels=readyToReceive(s,sourceModels,ID)
                    sourceModels,METASTATS=readyToReceive(s,sourceModels,UID,METASTATS)
                    # modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget = modelMultiHistory.nextModels(
                            # existingModels,transitionMatrix,modelOrder,partial,modelID,tLabel,DROP_FIELDS,startIDX,
                            # False,weightType,tempModel,DEFAULT_PRED)
                    conceptuallySimGroup = []
                    conceptSimPreds= pd.DataFrame()
                    initModelSet = modelSet.copy().keys()
                    existingModels,sourceModels,modelSet,distanceMatrix = removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix)
                    print("relearning becuase temp model: "+str(modelID))
                    print("tempModel flag: "+str(tempModel))
                    modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget,METASTATS = modelMultiHistory.nextModels(
                            existingModels,transitionMatrix,modelOrder,partial,modelID,tLabel,DROP_FIELDS,startIDX,False,
                            weightType,tempModel,PCAVAR,DEFAULT_PRED,LEARNER_TYPE,METASTATS)
                    print("picked or learnt new model becuase temp model: "+str(modelID))
                    if not set(initModelSet) == set(modelSet.keys()):
                        recluster = True
                    # if 'AWE' in weightType:
                        # modelSet = AWEaddToModelSet(partial,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS)
                    # elif 'AddExpO' in weightType:
                        # print(lastModID)
                        # print(modelSet)
                        # modelSet,orderedModels,weights['metaModel'] = AddExpOaddToModelSet(partial,idx,lastModID,existingModels,
                                # sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS)
                        # METASTATS['metaWeightUpdate']+=1
                        # print("after")
                        # print(modelSet)
                        # print(modelMultiHistory.isStable(lastModID,existingModels))
                    # elif 'AddExpP' in weightType:
                        # print(lastModID)
                        # print(modelSet)
                        # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(partial,idx,lastModID,existingModels,
                                # sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS)
                        # METASTATS['metaWeightUpdate']+=1
                        # print("after")
                        # print(modelSet)
                        # print(modelMultiHistory.isStable(lastModID,existingModels))
                        # # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(df,None,None,sourceModels,
                                # # modelSet,orderedModels,weights['metaModel'],tLabel,DROP_FIELDS)
                    # else:
                        # initModelSet = modelSet.copy().keys()
                        # modelSet,sourceModels,distanceMatrix,tc = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
                        # if not set(initModelSet) == set(modelSet.keys()):
                            # recluster = True
                        # METASTATS['totalPACalcs']+=tc
                        # if tc > 0:
                            # METASTATS['triggerPACalcs']+=1
                    # send to controller
                    # ofUse = 0
                    ofUse = existingModels[modelID]['usedFor']
                    sentFlag = 0
                    tempModel = False
                    # stableLocals = modelMultiHistory.getStableModels(existingModels)
                    # weights = modelMulti.calcWeights(partial,sourceModels,targetModel,tLabel,
                            # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                    # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(partial,sourceModels,
                            # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                            # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames)
                    if 'AddExp' not in weightType:
                        currentBases = []
                        if 'metaXColumns' in weights.keys():
                            currentBases = weights['metaXColumns'].tolist()
                        print("tempModel just discarded:"+str(currentBases))
                        weights,distanceMatrix,affinityMatrix,groupedNames,tc,METASTATS = modelMulti.calcWeights(partial,modelSet,
                                targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,recluster)
                        print("recluster is: "+str(recluster))
                        print("models used now are: "+str(weights['metaXColumns'].tolist()))
                        print("recalcWeights6")
                        recluster = False
                        # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(partial,modelSet,
                                # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                        if 'PAC' in weightType or 'CL2' in weightType:
                            METASTATS['totalPACalcs']+=tc[1]
                            if tc[1] > 0:
                                METASTATS['triggerPACalcs']+=1
                            METASTATS['totalOtherMetricCalcs']+=tc[0]
                            if tc[0]>0:
                                METASTATS['triggerMetricCalcs']+=1
                        else:
                            METASTATS['totalOtherMetricCalcs']+=tc
                            if tc>0:
                                METASTATS['triggerMetricCalcs']+=1
                        METASTATS['metaWeightUpdate']+=1
                    if 'AWE' in weightType:
                        modelSet = weights['AWEmodelSet']
                    newTarget = False
                    adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                    # window = createModel.initialPredict(window,targetModel,tLabel,DROP_FIELDS)
                    # if lastModID < modelID:
                        # drifts[modelCount] = week
                    
                    # startDate=idx
                    # modelID,existingModels,transitionMatrix,modelOrder,targetModel,sourceAlpha = gotlHistory.nextModels(existingModels,transitionMatrix,modelOrder,partial,
                                # modelID,sourceAlpha,DELTA,tLabel,DROP_FIELDS,startDate,False)
                    # #create new adwin
                    # adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                    retrainFlag=False
                    window = window.iloc[0:0]
                elif len(window) == int(MAX_WINDOW/2):
                    print("temp model created with: "+str(len(window))+str(tempModel))
                    startIDX=idx

                    # sourceModels=readyToReceive(s,sourceModels,ID)
                    # sourceModels,METASTATS=readyToReceive(s,sourceModels,UID,METASTATS)
                    # initModelSet = modelSet.copy().keys()
                    # existingModels,sourceModels,modelSet,distanceMatrix = removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix)

                    ##########NEED TO DEAL WITH THIS\\\\\\\\\\\\
                    conceptuallySimGroup = []
                    conceptSimPreds= pd.DataFrame()
                    modelID,existingModels,transitionMatrix,modelOrder,model,newTarget,METASTATS = modelMultiHistory.tempModel(existingModels,
                            transitionMatrix,modelOrder,window,modelID,tLabel,DROP_FIELDS,startIDX,False,tempModel,DEFAULT_PRED,LEARNER_TYPE,METASTATS)
                    # if not set(initModelSet) == set(modelSet.keys()):
                        # recluster = True
                    if 'AWE' in weightType:
                        modelSet,METASTATS = AWEaddToModelSet(window,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                    elif 'AddExpO' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpOaddToModelSet(window,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                    elif 'AddExpP' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpPaddToModelSet(window,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                        # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(df,None,None,sourceModels,
                                # modelSet,orderedModels,weights['metaModel'],tLabel,DROP_FIELDS)
                    else:
                        initModelSet = modelSet.copy().keys()
                        print("before 7, adding "+str(lastModID)+" to model set")
                        modelSet,sourceModels,distanceMatrix,tc,METASTATS = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,STABLE_SIZE,distanceMatrix,REPLACEMENT)
                        if not set(initModelSet) == set(modelSet.keys()):
                            recluster = True
                        METASTATS['totalPACalcs']+=tc
                        if tc > 0:
                            METASTATS['triggerPACalcs']+=1
                    # weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,
                            # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                    # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(window,sourceModels,
                            # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                            # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames)
                    if 'AddExp' not in weightType:
                        currentBases = []
                        if 'metaXColumns' in weights.keys():
                            currentBases = weights['metaXColumns'].tolist()
                            print("just about to recalc7:"+str(currentBases))
                        # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(window,modelSet,
                                # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,recluster)
                        weights,distanceMatrix,affinityMatrix,groupedNames,tc,METASTATS = modelMulti.calcWeights(window,modelSet,
                                targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                None,distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,False)
                        print("recalcWeights7")
                        print("conceptSimGroup:"+str(conceptuallySimGroup))
                        # recluster = False
                        # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(window,modelSet,
                                # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                        if 'PAC' in weightType or 'CL2' in weightType:
                            METASTATS['totalPACalcs']+=tc[1]
                            if tc[1] > 0:
                                METASTATS['triggerPACalcs']+=1
                            METASTATS['totalOtherMetricCalcs']+=tc[0]
                            if tc[0]>0:
                                METASTATS['triggerMetricCalcs']+=1
                        else:
                            METASTATS['totalOtherMetricCalcs']+=tc
                            if tc>0:
                                METASTATS['triggerMetricCalcs']+=1
                        METASTATS['metaWeightUpdate']+=1
                    if 'AWE' in weightType:
                        modelSet = weights['AWEmodelSet']
                    newTarget = False
                    ofUse = 0
                    if modelID in existingModels.keys():
                        ofUse = existingModels[modelID]['usedFor']
                    tempModel = True
                    adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
            if updateFlag:
                retrainFlag = True
                print("change has been detected at: "+str(idx))
                print("n1, n0: "+str(n1)+", "+str(n0))
                print("u1, u0: "+str(u1)+", "+str(u0))
                
                print(len(conceptSimPreds))
                conceptSimPreds = conceptSimPreds.iloc[:-1*int(n1),]
                print(len(conceptSimPreds))
                # if conceptuallySimGroup and len(conceptSimPreds)>0:
                if REPLACEMENT and conceptuallySimGroup and len(conceptSimPreds)>=MAX_WINDOW:
                    maxModelID = evalConceptSimilar(modelID,conceptSimPreds,tLabel)
                    if maxModelID != modelID:
                        print(maxModelID)
                        print("need to sub")
                        substituteModelID = maxModelID
                        if substituteModelID in modelSet.keys():
                            substituteModelInfo = modelSet[substituteModelID]
                        else:
                            substituteModelInfo = stableLocals[substituteModelID]
                        # substituteModelInfo = modelSet[substituteModelID]
                        # print("existingModels")
                        # print(existingModels)
                        print("transitionmatrix")
                        print(transitionMatrix)
                        existingModels,transitionMatrix = modelMultiHistory.substituteModel(existingModels,transitionMatrix,modelID,substituteModelID,substituteModelInfo)
                        # print("existingModels")
                        # print(existingModels)
                        print("transitionMatrix")
                        print(transitionMatrix)
                        if uniqueModID(modelID,UID) in modelsSent or len(delayedTransferList)>0:# or substituteModelID not in modelsSent:
                            replacementFlag,modelsSent,METASTATS = replacementModelTransfer(UID,modelID,substituteModelID,substituteModelInfo,s,modelsSent,NEIGHBORS,METASTATS)
                        METASTATS['COMMSTATS']['NumModelsNotSentDelayed']+=len(delayedTransferList)
                        delayedTransferList = []
                        initModelSet = modelSet.copy().keys()
                        modelSet,distanceMatrix = updateModelSetDM(modelID,substituteModelID,modelSet,existingModels,distanceMatrix)
                        if not set(initModelSet) == set(modelSet.keys()):
                            recluster = True
                    else:
                        print("need to send model eventually")
                        print("send to: "+str(delayedTransferList))
                        print(conceptuallySimGroup)
                        print(maxModelID)
                        print(modelID)
                        print(delayedTransferList)
                        if len(delayedTransferList)>0:
                            modelInfoToSend = {'model':targetModel,'PCs':existingModels[modelID]['PCs']}
                            delaySentFlag,modelsSent,METASTATS = delayedModelTransfer(UID,modelID,modelInfoToSend,s,modelsSent,delayedTransferList,METASTATS)
                            if delaySentFlag:
                                METASTATS['LOCALMODELSTATS']['modelsDelayed']+=1
                                METASTATS['LOCALMODELSTATS']['modelsDelayedIDs'].append(uniqueModID(modelID,UID))
                                delayedTransferList = []
                        
                conceptuallySimGroup = []
                conceptSimPreds= pd.DataFrame()
                conceptSize = 0
                
                if n1 >= MAX_WINDOW:
                    lastModID = modelID
                    modelCount+=1
                    startIDX = idx
                    modelStart.append(startIDX)
                    conceptSize = len(window)
                    # partial = window.copy()
                    partial = historyData.iloc[(MAX_WINDOW*(-1)):].copy()
                    #retrain model
                    # sourceModels=readyToReceive(s,sourceModels,ID)
                    conceptuallySimGroup = []
                    conceptSimPreds= pd.DataFrame()
                    sourceModels,METASTATS=readyToReceive(s,sourceModels,UID,METASTATS)
                    initModelSet = modelSet.copy().keys()
                    existingModels,sourceModels,modelSet,distanceMatrix = removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix)
                    modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget,METASTATS = modelMultiHistory.nextModels(
                            existingModels,transitionMatrix,modelOrder,partial,modelID,tLabel,DROP_FIELDS,startIDX,False,
                            weightType,tempModel,PCAVAR,DEFAULT_PRED,LEARNER_TYPE,METASTATS)
                    if not set(initModelSet) == set(modelSet.keys()):
                        recluster = True
                    tempModel = False
                    if 'AWE' in weightType:
                        modelSet,METASTATS = AWEaddToModelSet(partial,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                    elif 'AddExpO' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpOaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                    elif 'AddExpP' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpPaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                        # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(df,None,None,sourceModels,
                                # modelSet,orderedModels,weights['metaModel'],tLabel,DROP_FIELDS)
                    else:
                        initModelSet = modelSet.copy().keys()
                        # modelSet,sourceModels,distanceMatrix,tc = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
                        modelSet,sourceModels,distanceMatrix,tc,METASTATS = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,STABLE_SIZE,distanceMatrix,REPLACEMENT)
                        if not set(initModelSet) == set(modelSet.keys()):
                            recluster = True
                        METASTATS['totalPACalcs']+=tc
                        if tc > 0:
                            METASTATS['triggerPACalcs']+=1
                    # send to controller
                    # ofUse = 0
                    ofUse = existingModels[modelID]['usedFor']
                    sentFlag = 0
                    # stableLocals = modelMultiHistory.getStableModels(existingModels)
                    # weights = modelMulti.calcWeights(partial,sourceModels,targetModel,tLabel,
                            # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                    # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(partial,sourceModels,
                            # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                            # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames)
                    if 'AddExp' not in weightType:
                        currentBases = []
                        if 'metaXColumns' in weights.keys():
                            currentBases = weights['metaXColumns'].tolist()
                        weights,distanceMatrix,affinityMatrix,groupedNames,tc,METASTATS = modelMulti.calcWeights(partial,modelSet,
                                targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,recluster)
                        print("recalcWeights1")
                        recluster = False
                        # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(partial,modelSet,
                                # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                        if 'PAC' in weightType or 'CL2' in weightType:
                            METASTATS['totalPACalcs']+=tc[1]
                            if tc[1] > 0:
                                METASTATS['triggerPACalcs']+=1
                            METASTATS['totalOtherMetricCalcs']+=tc[0]
                            if tc[0]>0:
                                METASTATS['triggerMetricCalcs']+=1
                        else:
                            METASTATS['totalOtherMetricCalcs']+=tc
                            if tc>0:
                                METASTATS['triggerMetricCalcs']+=1
                        METASTATS['metaWeightUpdate']+=1
                    if 'AWE' in weightType:
                        modelSet = weights['AWEmodelSet']
                    newTarget = False
                    adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                    # if lastModID < modelID:
                        # drifts[modelCount] = week
                    
                    retrainFlag=False
                    window = window.iloc[0:0]
                else:
                    window = historyData.iloc[int(n1)*-1:].copy()
                    lastModID = modelID
                    if 'AWE' in weightType:
                        modelSet,METASTATS = AWEaddToModelSet(partial,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                    elif 'AddExpO' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpOaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                    elif 'AddExpP' in weightType:
                        print(lastModID)
                        print(modelSet)
                        modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpPaddToModelSet(partial,idx,lastModID,existingModels,
                                sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                        METASTATS['metaWeightUpdate']+=1
                        print("after")
                        print(modelSet)
                        print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                        # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(df,None,None,sourceModels,
                                # modelSet,orderedModels,weights['metaModel'],tLabel,DROP_FIELDS)
                    else:
                        initModelSet = modelSet.copy().keys()
                        # modelSet,sourceModels,distanceMatrix,tc = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
                        modelSet,sourceModels,distanceMatrix,tc,METASTATS = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,STABLE_SIZE,distanceMatrix,REPLACEMENT)
                        if not set(initModelSet) == set(modelSet.keys()):
                            recluster = True
                        METASTATS['totalPACalcs']+=tc
                        if tc > 0:
                            METASTATS['triggerPACalcs']+=1
                    recluster = False
                    # if existingModels[modelID]['substituted']==True:
                        # lastModID = existingModels[modelID]['subID']
                    # else:
                        # lastModID = modelID
                    # lastModID = modelID
                    
                    if n1 >= MAX_WINDOW/2:
                        if modelMultiHistory.reuseModels(existingModels,window, tLabel,DROP_FIELDS):
                            startIDX=idx
                            # print("22: "+str(len(window))+str(tempModel))
                            # sourceModels=readyToReceive(s,sourceModels,ID)
                            sourceModels,METASTATS=readyToReceive(s,sourceModels,UID,METASTATS)
                            # modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget = modelMultiHistory.nextModels(
                                    # existingModels,transitionMatrix,modelOrder,window,modelID,tLabel,DROP_FIELDS,startIDX,
                                    # False,weightType,tempModel,DEFAULT_PRED)
                            conceptuallySimGroup = []
                            conceptSimPreds= pd.DataFrame()
                            initModelSet = modelSet.copy().keys()
                            existingModels,sourceModels,modelSet,distanceMatrix = removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix)
                            modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget,METASTATS = modelMultiHistory.nextModels(
                                    existingModels,transitionMatrix,modelOrder,window,modelID,tLabel,DROP_FIELDS,startIDX,False,
                                    weightType,tempModel,PCAVAR,DEFAULT_PRED,LEARNER_TYPE,METASTATS)
                            if not set(initModelSet) == set(modelSet.keys()):
                                recluster = True
                            if 'AWE' in weightType:
                                modelSet,METASTATS = AWEaddToModelSet(window,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                            elif 'AddExpO' in weightType:
                                print(lastModID)
                                print(modelSet)
                                modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpOaddToModelSet(window,idx,lastModID,existingModels,
                                        sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                                METASTATS['metaWeightUpdate']+=1
                                print("after")
                                print(modelSet)
                                print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                            elif 'AddExpP' in weightType:
                                print(lastModID)
                                print(modelSet)
                                modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpPaddToModelSet(window,idx,lastModID,existingModels,
                                        sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                                METASTATS['metaWeightUpdate']+=1
                                print("after")
                                print(modelSet)
                                print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                                # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(df,None,None,sourceModels,
                                        # modelSet,orderedModels,weights['metaModel'],tLabel,DROP_FIELDS)
                            else:
                                initModelSet = modelSet.copy().keys()
                                # modelSet,sourceModels,distanceMatrix,tc = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,distanceMatrix)
                                modelSet,sourceModels,distanceMatrix,tc,METASTATS = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,STABLE_SIZE,distanceMatrix,REPLACEMENT)
                                if not set(initModelSet) == set(modelSet.keys()):
                                    recluster = True
                                METASTATS['totalPACalcs']+=tc
                                if tc > 0:
                                    METASTATS['triggerPACalcs']+=1
                            # ofUse = 0
                            ofUse = existingModels[modelID]['usedFor']
                            sentFlag = 0
                            # weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,
                                    # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                            # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(window,sourceModels,
                                    # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                    # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames)
                            if 'AddExp' not in weightType:
                                currentBases = []
                                if 'metaXColumns' in weights.keys():
                                    currentBases = weights['metaXColumns'].tolist()
                                weights,distanceMatrix,affinityMatrix,groupedNames,tc,METASTATS = modelMulti.calcWeights(window,modelSet,
                                        targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                        existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,recluster)
                                print("recalcWeights2")
                                recluster = False
                                # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(window,modelSet,
                                        # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                        # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                                if 'PAC' in weightType or 'CL2' in weightType:
                                    METASTATS['totalPACalcs']+=tc[1]
                                    if tc[1] > 0:
                                        METASTATS['triggerPACalcs']+=1
                                    METASTATS['totalOtherMetricCalcs']+=tc[0]
                                    if tc[0]>0:
                                        METASTATS['triggerMetricCalcs']+=1
                                else:
                                    METASTATS['totalOtherMetricCalcs']+=tc
                                    if tc>0:
                                        METASTATS['triggerMetricCalcs']+=1
                                METASTATS['metaWeightUpdate']+=1
                            if 'AWE' in weightType:
                                modelSet = weights['AWEmodelSet']
                            newTarget = False
                            tempModel = False
                            adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                            print("old model and new adwin windows created")
                            retrainFlag = False
                            window = window.iloc[0:0]
                        else: 
                            # if n1 >= MAX_WINDOW/2:
                            # if existingModels[modelID]['substituted']==True:
                                # lastModID = existingModels[modelID]['subID']
                            # else:
                                # lastModID = modelID
                            lastModID = modelID
                            # print("temp model created with: "+str(len(window))+str(tempModel))
                            # print(tempModel)
                            startIDX=idx
                            # print("3: "+str(len(window))+str(tempModel))
                            # sourceModels=readyToReceive(s,sourceModels,ID)
                            # sourceModels,METASTATS=readyToReceive(s,sourceModels,UID,METASTATS)
                            #############NED TO FIX\\\\\\\\\\\\\\\\
                            conceptuallySimGroup = []
                            conceptSimPreds= pd.DataFrame()
                            # conceptuallySimGroup = []
                            # conceptSimPreds= pd.DataFrame()
                            # initModelSet = modelSet.copy().keys()
                            # existingModels,sourceModels,modelSet,distanceMatrix = removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix)
                            modelID, existingModels,transitionMatrix,modelOrder,targetModel,newTarget,METASTATS = modelMultiHistory.tempModel(
                                    existingModels,transitionMatrix,modelOrder,window,modelID,tLabel,DROP_FIELDS,startIDX,
                                    False,tempModel,DEFAULT_PRED,LEARNER_TYPE,METASTATS)
                            # if not set(initModelSet) == set(modelSet.keys()):
                                # recluster = True
                            # weights = modelMulti.calcWeights(window,sourceModels,targetModel,tLabel,
                                    # DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,existingModels[modelID]['PCs'])
                            # weights,distanceMatrix,affinityMatrix,groupedNames = modelMulti.calcWeights(window,sourceModels,
                                    # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                    # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames)
                            if 'AWE' in weightType:
                                modelSet,METASTATS = AWEaddToModelSet(partial,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                            elif 'AddExpO' in weightType:
                                print(lastModID)
                                print(modelSet)
                                modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpOaddToModelSet(partial,idx,lastModID,existingModels,
                                        sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                                METASTATS['metaWeightUpdate']+=1
                                print("after")
                                print(modelSet)
                                print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                            elif 'AddExpP' in weightType:
                                print(lastModID)
                                print(modelSet)
                                modelSet,orderedModels,weights['metaModel'],METASTATS = AddExpPaddToModelSet(partial,idx,lastModID,existingModels,
                                        sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,STABLE_SIZE)
                                METASTATS['metaWeightUpdate']+=1
                                print("after")
                                print(modelSet)
                                print(modelMultiHistory.isStable(lastModID,existingModels,STABLE_SIZE))
                                # modelSet,orderedModels,weights['metaModel'] = AddExpPaddToModelSet(df,None,None,sourceModels,
                                        # modelSet,orderedModels,weights['metaModel'],tLabel,DROP_FIELDS)
                            else:
                                initModelSet = modelSet.copy().keys()
                                modelSet,sourceModels,distanceMatrix,tc,METASTATS = addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,STABLE_SIZE,distanceMatrix,REPLACEMENT)
                                if not set(initModelSet) == set(modelSet.keys()):
                                    recluster = True
                                METASTATS['totalPACalcs']+=tc
                                if tc > 0:
                                    METASTATS['triggerPACalcs']+=1
                            
                            if 'AddExp' not in weightType:
                                currentBases = []
                                if 'metaXColumns' in weights.keys():
                                    currentBases = weights['metaXColumns'].tolist()
                                stableLocals = modelMultiHistory.getStableModels(existingModels)
                                weights,distanceMatrix,affinityMatrix,groupedNames,tc,METASTATS = modelMulti.calcWeights(window,modelSet,
                                        targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                        None,distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,False)
                                # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(window,modelSet,
                                        # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                        # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,recluster)
                                print("recalcWeights3")
                                # recluster = False
                                # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(window,modelSet,
                                        # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                        # existingModels[modelID]['PCs'],distanceMatrix,affinityMatrix,newTarget,groupedNames,METASTATS['currentBaseLearnerIDs'])
                                if 'PAC' in weightType or 'CL2' in weightType:
                                    METASTATS['totalPACalcs']+=tc[1]
                                    if tc[1] > 0:
                                        METASTATS['triggerPACalcs']+=1
                                    METASTATS['totalOtherMetricCalcs']+=tc[0]
                                    if tc[0]>0:
                                        METASTATS['triggerMetricCalcs']+=1
                                else:
                                    METASTATS['totalOtherMetricCalcs']+=tc
                                    if tc>0:
                                        METASTATS['triggerMetricCalcs']+=1
                                METASTATS['metaWeightUpdate']+=1
                            if 'AWE' in weightType:
                                modelSet = weights['AWEmodelSet']
                            newTarget = False
                            ofUse = 0
                            if modelID in existingModels.keys():
                                ofUse = existingModels[modelID]['usedFor']
                            tempModel = True
                            adwin=Adwin(MAX_WINDOW,ADWIN_DELTA)
                        # else:
                            # if 'AddExp' not in weightType:
                                # currentBases = []
                                # if 'metaXColumns' in weights.keys():
                                    # currentBases = weights['metaXColumns'].tolist()
                                # if existingModels[modelID]['substituted']==True:
                                    # if modelID in currentBases:
                                        # currentBases.remove(modelID)
                                    # modelID = existingModels[modelID]['subID']
                                    # if modelID not in currentBases:
                                        # currentBases.append(modelID)
                                # weights,distanceMatrix,affinityMatrix,groupedNames,tc = modelMulti.calcWeights(window,modelSet,
                                        # targetModel,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH,stableLocals,modelID,
                                        # None,distanceMatrix,affinityMatrix,newTarget,groupedNames,currentBases,False)
                                    # # weights['metaXColumns']=currentBases
                                # print("current bases are: "+str(currentBases))
                                # print("metaX cols are: "+str(weights['metaXColumns'].tolist()))
                                
                                # ofUse = 0
        # if idx%75 == (startDoW-1) and storeFlag == True:
        for b in METASTATS['currentBaseLearnerIDs']:
            if b in METASTATS['modelsReceived']:
                if b not in METASTATS['baseUsage'].keys():
                    METASTATS['baseUsage'][b]=1
                else:
                    METASTATS['baseUsage'][b]+=1
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
    print(modelsSent)
    # for i in modelsSent:
        # print(i)
        # print(existingModels[i]['usedFor'])
    print(modelOrder)
    print(existingModels)
    print(transitionMatrix)
    print(modelStart)
    with pd.option_context('display.max_rows',None):
        print(historyData[['predictions',tLabel]])


    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed



def simTarget(weightType,s):
    global METASTATS
    sourceModels = dict()
    sourceModels,METASTATS = readyToReceive(s,sourceModels,UID,METASTATS)
    # sourceModels = dict()
    # sourceModels = readyToReceive(s,sourceModels,ID)

    #get models from controller
    print("source Models:")
    print(sourceModels)
    if DEFAULT_PRED == 'Following':
        df = preprocess.pullFollowingData(FP)
        df = preprocess.subsetFollowingData(df,SIM_FROM,SIM_TO)
    elif DEFAULT_PRED == 'Heating':
        df = preprocess.pullHeatingData(FP)
        df = preprocess.subsetHeatingData(df,SIM_FROM,SIM_TO)
    else:
        df = preprocess.pullData(FP)
        df = preprocess.subsetData(df,SIM_FROM,SIM_TO)
    df['predictions'] = np.nan
    
    historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights,p,numModelsUsed = runAWPro(df,sourceModels,tLabel,weightType,s)
    
    numModelsUsedArr = np.array(numModelsUsed)
    meanNumModels = numModelsUsedArr.mean()
    print("stablesixe: "+str(STABLE_SIZE))
    print(" maxwindow: "+str(MAX_WINDOW))
    print("adwinDelta: "+str(ADWIN_DELTA))
    if WEIGHTTYPE == 'OLS':
        fileWeight = 'OLS'
    elif WEIGHTTYPE == 'OLSFEMI':
        fileWeight = str(WEIGHTTYPE)+str(MITHRESH)+"_"+str(CULLTHRESH)
    elif WEIGHTTYPE == 'OLSFEMIRed':
        fileWeight = str(WEIGHTTYPE)+str(MITHRESH)+"_"+str(CULLTHRESH)
    elif WEIGHTTYPE == 'OLSFEMIPARed':
        fileWeight = str(WEIGHTTYPE)+str(MITHRESH)+"_"+str(CULLTHRESH)
    elif WEIGHTTYPE == 'OLSFEPA':
        fileWeight = str(WEIGHTTYPE)+str(PATHRESH)+"_"+str(CULLTHRESH)
    elif WEIGHTTYPE == 'OLSPARed':
        fileWeight = 'OLSPARed'
    elif WEIGHTTYPE == 'OLSKPAC2Red':
        fileWeight = 'OLSKPAC2Red'
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
    elif WEIGHTTYPE == 'OLSAWE':
        fileWeight = 'OLSAWE'
    elif WEIGHTTYPE == 'OLSAddExpO':
        fileWeight = 'OLSAddExpO'
    elif WEIGHTTYPE == 'OLSAddExpP':
        fileWeight = 'OLSAddExpP'
    else:
        fileWeight = str(WEIGHTTYPE)+str(CULLTHRESH)
    
    if not os.path.exists('Results/AWPro/'+str(DEFAULT_PRED)):
        os.mkdir('Results/AWPro/'+str(DEFAULT_PRED))
    if not os.path.exists('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)):
        os.mkdir('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE))
    if not os.path.exists('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight)):
        os.mkdir('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight))
    if not os.path.isfile('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight)+'/'+str(REPLACEMENTSTR)+'results'+str(NUMSTREAMS)+"streams.csv"):
        resFile = open('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight)+'/'+str(REPLACEMENTSTR)+'results'+str(NUMSTREAMS)+"streams.csv",'a')
        # if not os.path.isfile('Results5000/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv"):
        # resFile = open('Results5000/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
        # resFile.write('sID,R2,RMSE,Pearsons,PearsonsPval,Err,numModels,reproR2,reproRMSE,reproPearsons,reproPearsonsPval,reproErr,numModelsUsed\n')
        # resFile.write('sID,Win,Thresh,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,'+
                # 'reproErr,reproPearsons,reproPearsonsPval,numModelsUsed,totalLocalModels,totalStableLocalModels,'+
                # 'METAnumModelsSent,METAstableLocals,METAnumModelsRec,METABaseChange,METABaseUsage,METAAvgBases,METAWeightUpdates\n')
        # resFile.write('sID,Win,AdwinDelta,acc,prob,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,'+
                # 'reproErr,reproPearsons,reproPearsonsPval,conn_deg,neighbors,numModelsUsed,totalLocalModels,totalStableLocalModels,'+
                # 'METAnumModelsSent,METAstableLocals,METAnumModelsRec,METABaseChange,METABaseUsage,METAAvgBases,METAWeightUpdates,'+
                # 'totalPACalcs,totalOtherMetricCalcs,triggerPACalcs,triggerMetricCalcs\n')
        headerSTR = 'sID,Win,AdwinDelta,acc,prob,'
        headerSTR = genResultsHeader(headerSTR)
        resFile.write(headerSTR)
    else:
        # resFile = open('Results5000/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
        resFile = open('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(FRAMEWORKTYPE)+'/'+str(fileWeight)+'/'+str(REPLACEMENTSTR)+'results'+str(NUMSTREAMS)+"streams.csv",'a')
    instanceResults =(str(UID)+','+str(MAX_WINDOW)+','+str(ADWIN_DELTA)+','+
        str(MODEL_HIST_THRESHOLD_ACC)+','+str(MODEL_HIST_THRESHOLD_PROB)+','+str(CONN_DEG)+','+
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
        str(pearsonr(p['predictions'].values,p[tLabel].values)[1])+',')
        # str(CONN_DEG)+','+'\"'+str(NEIGHBORS)+'\",'+
    instanceResults = genResultsInstance(instanceResults,METASTATS)
    instanceResults = instanceResults+(str(meanNumModels)+','+str(METASTATS['baseLearnerChange'])+','+
        str(sum(METASTATS['numberBaseLearners'])/len(METASTATS['numberBaseLearners']))+','+
        str(METASTATS['metaWeightUpdate'])+','+'\"'+str(NEIGHBORS)+'\",'+
        '\"'+str(METASTATS['baseUsage'])+'\"'+'\n')
    resFile.write(instanceResults)
    # resFile.write(str(UID)+','+str(MAX_WINDOW)+','+str(ADWIN_DELTA)+','+
            # str(MODEL_HIST_THRESHOLD_ACC)+','+str(MODEL_HIST_THRESHOLD_PROB)+','+
            # str(metrics.r2_score(historyData[tLabel],historyData['predictions']))+','+
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
            # str(CONN_DEG)+','+'\"'+str(NEIGHBORS)+'\",'+
            # str(meanNumModels)+','+str(len(existingModels))+','+
            # str(len(modelMultiHistory.getStableModels(existingModels)))+','+
            # str(len(METASTATS['modelsSent']))+','+str(len(METASTATS['stableLocal']))+','+
            # str(len(METASTATS['modelsReceived']))+','+str(METASTATS['baseLearnerChange'])+','+
            # '\"'+str(METASTATS['baseUsage'])+'\",'+
            # str(sum(METASTATS['numberBaseLearners'])/len(METASTATS['numberBaseLearners']))+','+
            # str(METASTATS['metaWeightUpdate'])+','+
            # str(METASTATS['totalPACalcs'])+','+
            # str(METASTATS['totalOtherMetricCalcs'])+','+
            # str(METASTATS['triggerPACalcs'])+','+
            # str(METASTATS['triggerMetricCalcs'])+'\n')
            # str(len(modelMultiHistory.getStableModels(existingModels)))+','+
            # str(len(METASTATS['modelsSent']))+','+str(len(METASTATS['stableLocal']))+','+
            # str(len(METASTATS['modelsReceived']))+','+str(METASTATS['baseLearnerChange'])+','+
            # '\"'+str(METASTATS['baseUsage'])+'\",'+
            # str(sum(METASTATS['numberBaseLearners'])/len(METASTATS['numberBaseLearners']))+','+
            # str(METASTATS['metaWeightUpdate'])+'\n')

    # resFile.write("METASTATS stable local: "+str(METASTATS['stableLocal'])+'\n'+
        # "METASTATS modelsSent: "+str(METASTATS['modelsSent'])+'\n'+
        # "METASTATS modelsReceived: "+str(METASTATS['modelsReceived'])+'\n'+
        # "METASTATS baseLearnerChange: "+str(METASTATS['baseLearnerChange'])+'\n'+
        # "METASTATS baseLearnerList: "+str(METASTATS['baseLearnerList'])+'\n'+
        # "METASTATS baseUsage: "+str(METASTATS['baseUsage'])+'\n'+
        # "METASTATS numberBaseLearners: "+str(METASTATS['numberBaseLearners'])+'\n'+
        # "METASTATS totalNumInstances: "+str(METASTATS['totalNumInstances'])+'\n'+
        # "\n\nAVERAGING\n\n"+
        # "total number models sent: "+str(len(METASTATS['modelsSent']))+'\n'+
        # "total number models stable: "+str(len(METASTATS['stableLocal']))+'\n'+
        # "total number models received: "+str(len(METASTATS['modelsReceived']))+'\n'+
        # "total average num base learners: "+str(sum(METASTATS['numberBaseLearners'])/len(METASTATS['numberBaseLearners']))+'\n'+
        # "number of nonused models: "+str(sum(b==0 for b in METASTATS['baseUsage'].values()))+'\n'+
        # "total weight updates: "+str(METASTATS['metaWeightUpdate'])+'\n'+
        # "PERFORMANCE: "+ str(metrics.r2_score(historyData[tLabel],historyData['predictions'])))
    resFile.close()

    # if not os.path.exists('Results/AWPro/'+str(DEFAULT_PRED)):
        # os.mkdir('Results/AWPro/'+str(DEFAULT_PRED))
    # if not os.path.exists('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(fileWeight)):
        # os.mkdir('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(fileWeight))
    # if not os.path.isfile('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv"):
        # resFile = open('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
        # resFile.write('sID,Win,AdwinDelta,acc,prob,R2,RMSE,Err,Pearsons,PearsonsPval,numModels,reproR2,reproRMSE,reproErr,reproPearsons,reproPearsonsPval,numModelsUsed,totalLocalModels,totalStableLocalModels\n')
    # else:
        # resFile = open('Results/AWPro/'+str(DEFAULT_PRED)+'/'+str(fileWeight)+'/results'+str(NUMSTREAMS)+"streams.csv",'a')
    # resFile.write(str(UID)+','+str(MAX_WINDOW)+','+str(ADWIN_DELTA)+','+
            # str(MODEL_HIST_THRESHOLD_ACC)+','+str(MODEL_HIST_THRESHOLD_PROB)+','+
            # str(metrics.r2_score(historyData[tLabel],historyData['predictions']))+','+
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
            # str(meanNumModels)+','+str(len(existingModels))+','+
            # str(len(modelMultiHistory.getStableModels(existingModels)))+'\n')

    # resFile.close()


    return historyData,modelStart,existingModels,transitionMatrix,modelOrder,drifts,multiWeights

def main():
    global ID
    global PORT
    global SIM_FROM
    global SIM_TO
    global FP
    global INIT_DAYS
    global STABLE_SIZE
    global MAX_WINDOW
    global ADWIN_DELTA
    global RUNID
    global NUMSTREAMS
    global WEIGHTTYPE
    global UID
    global CULLTHRESH
    global MITHRESH
    global DEFAULT_PRED
    global DROP_FIELDS
    global tLabel
    global EPSILON
    global PATHRESH
    global PCAVAR
    global MODEL_HIST_THRESHOLD_ACC
    global MODEL_HIST_THRESHOLD_PROB
    global METASTATS
    global LEARNER_TYPE
    global FRAMEWORKTYPE
    global CONN_DEG
    global NEIGHBORS
    global REPLACEMENT
    global REPLACEMENTSTR

    #loqxy
    parser = OptionParser(usage="usage: prog options",version="BOTL v2.0")
    parser.add_option("-k","--id",default="source0",dest="ID",help="name of source: sourceID")
    parser.add_option("-u","--uid",default="0",dest="UID",help="ID number of source")
    parser.add_option("-b","--port",default="3000",dest="PORT",help="PORT")
    parser.add_option("-f","--fp",default="./",dest="FP",help="file path")
    parser.add_option("-g","--from",default="0",dest="FROM",help="simulation from")
    parser.add_option("-j","--to",default="15000",dest="TO",help="simulation to")
    parser.add_option("-d","--domain",default = "Following",dest="DEFAULT_PRED",help="domain: Following, Heating, Sudden, Gradual")
    # parser.add_option("-t","--type",default = "RePro",dest= "CD_TYPE",help="Concept Drift Type: RePro, ADWIN, AWPro")
    parser.add_option("-w","--window",default = "90",dest="MAX_WINDOW",help="Window size (default = 90)")
    parser.add_option("-r","--ReProAcc",default = "0.5",dest="MODEL_HIST_THRESHOLD_ACC",help="RePro drift threshold")
    parser.add_option("-p","--ReProProb",default = "0.5",dest="MODEL_HIST_THRESHOLD_PROB",help="RePro recur prob")
    parser.add_option("-i","--runid",default = "1",dest="runID",help="RunID")
    parser.add_option("-n","--numStreams",default = "1",dest="numStreams",help="Number of streams")
    parser.add_option("-z","--ADWINDelta",default = "0.02",dest="ADWIN_DELTA",help="ADWIN confidence value")
    # parser.add_option("-e","--ReProThresh",default = "0.1",dest="THRESHOLD",help="RePro error threshold")
    parser.add_option("-s","--socket",default = "3000",dest="socketOffset",help="Socket Offset")
    parser.add_option("-e","--ensemble",default = "OLS",dest="weightType",help="Weight Type (OLS, OLSFE, OLSFEMI,...)")
    parser.add_option("-c","--perfCull",default = "0.0",dest="CThresh",help="Performance culling parameter")
    parser.add_option("-m","--miCull",default = "2",dest="MThresh",help="Mutual Information culling parameter")
    parser.add_option("-a","--paCull",default = "1",dest="PAThresh",help="Principal Angle culling parameter")
    parser.add_option("-v","--variance",default = "0.05",dest="PCAVar",help="Keep prinicpal components that capture this uch variance")
    parser.add_option("-l","--learner",default = "SVR",dest="learnerType",help="Type of model used by the underlying CDD")
    parser.add_option("-o","--frameworkType",default = "",dest="frameworkType",help="framework Type descriptor for results files")
    parser.add_option("--connDeg",default="0",dest="connDeg",help="degree of Connectivity")
    parser.add_option("--neighbors",default="",dest="neighborStr",help="list of neighbors for communication")
    parser.add_option("--replace",default="0",dest="replace",help="0: models shouldn't be replaced, 1: replace conceptually similar models")
    (options,args) = parser.parse_args()
    
    HOST = 'localhost'
    ID = str(options.ID)
    UID = str(options.UID)
    PORT = int(options.PORT)
    FP = str(options.FP)
    DEFAULT_PRED=str(options.DEFAULT_PRED)
    # CD_TYPE=str(options.CD_TYPE)
    MAX_WINDOW=int(options.MAX_WINDOW)
    MODEL_HIST_THRESHOLD_ACC=float(options.MODEL_HIST_THRESHOLD_ACC)
    MODEL_HIST_THRESHOLD_PROB=float(options.MODEL_HIST_THRESHOLD_PROB)
    RUNID=int(options.runID)
    NUMSTREAMS=int(options.numStreams)
    ADWIN_DELTA=float(options.ADWIN_DELTA)
    socketOffset=int(options.socketOffset)
    WEIGHTTYPE=str(options.weightType)
    CULLTHRESH=float(options.CThresh)
    MITHRESH=float(options.MThresh)
    PATHRESH=float(options.PAThresh)
    PCAVAR=float(options.PCAVar)
    LEARNER_TYPE=str(options.learnerType)
    FRAMEWORKTYPE = str(options.frameworkType)
    CONN_DEG = int(options.connDeg)
    NEIGHBORS = [str(n) for n in str(options.neighborStr).split(',')]
    THRESHOLD = MODEL_HIST_THRESHOLD_ACC
    STABLE_SIZE = MAX_WINDOW * 2
    INIT_DAYS = MAX_WINDOW
    REPLACEMENT = int(options.replace)
    if REPLACEMENT == 1:
        REPLACEMENTSTR = 'replace'

    if DEFAULT_PRED == 'Heating':
        SIM_FROM = datetime.strptime(options.FROM,"%Y-%m-%d").date()
        SIM_TO = datetime.strptime(options.TO,"%Y-%m-%d").date()
    else:
        SIM_FROM = int(options.FROM)
        SIM_TO = int(options.TO)

    if DEFAULT_PRED == 'Following':
        DROP_FIELDS = ['Time','predictions']
        tLabel='FollowingTime'
        EPSILON = 0.1
    elif DEFAULT_PRED == 'Heating':
        DROP_FIELDS = ['date','year','month','day','predictions','heatingOn']
        tLabel='dTemp'
        EPSILON = 0.5
    else:
        DROP_FIELDS = ['predictions']
        tLabel='y'
        EPSILON = 0.01
    
    '''
    # set up connection to controller #
    '''
    # METASTATS = {'stableLocal':list(),'modelsSent':list(),'modelsReceived':list(),'metaWeightUpdate':0,'baseLearnerChange':0,
            # 'baseLearnerList':list(),'currentBaseLearnerIDs':list(),'baseUsage':dict(),'numberBaseLearners':list(),'totalNumInstances':0}
    # METASTATS = {'stableLocal':list(),'modelsSent':list(),'modelsReceived':list(),'metaWeightUpdate':0,'baseLearnerChange':0,
            # 'baseLearnerList':list(),'currentBaseLearnerIDs':list(),'baseUsage':dict(),'numberBaseLearners':list(),'totalNumInstances':0,
            # 'totalPACalcs':0,'totalOtherMetricCalcs':0,'triggerPACalcs':0,'triggerMetricCalcs':0}
    METASTATS = {'stableLocal':list(),'modelsSent':list(),'modelsReceived':list(),'metaWeightUpdate':0,'baseLearnerChange':0,
            'baseLearnerList':list(),'currentBaseLearnerIDs':list(),'baseUsage':dict(),'numberBaseLearners':list(),'totalNumInstances':0,
            'totalPACalcs':0,'totalOtherMetricCalcs':0,'triggerPACalcs':0,'triggerMetricCalcs':0,'evalModelSet':0,'sizeModelSet':[],
            'COMPSTATS':dict(), 'COMMSTATS':dict(),'TRANSMODELSTATS':dict(),'LOCALMODELSTATS':dict(),'REPLACEMENTMODELSTATS':dict(),
            'OVERVIEWSTATS':dict()}

    METASTATS['TRANSMODELSTATS'] = {'modelsReceived':0,'modelsDiscarded':0,'receivedModelIDs':[],'discardedModelIDs':[]}
    METASTATS['LOCALMODELSTATS'] = {'modelsLearnt':0,'stableModelsLearnt':0,'modelsTransferred':0,'modelsDelayed':0,'stableNeverSent':0,
            'stableNeverSentIDs':[],'stableModelIDs':[],'transferredModelIDs':[],'modelsDelayedIDs':[]}
    METASTATS['REPLACEMENTMODELSTATS'] = {'modelsReplaced':0,'replacedWithLocal':0,'replacedWithTrans':0,'replacedModelIDs':[],'replacementModelIDs':[]}
    METASTATS['COMMSTATS'] = {'CSMList':0,'CSMListBytes':0,'ReplaceList':0,'DiscardList':0,'DiscardListBytes':0,
            'NumModelsSent':0,'NumModelsSentBytes':0,'NumUniqueModelsSent':0,'NumUniqueModelsSentBytes':0,'NumModelsRec':0,'NumModelsRecBytes':0,'NumModelsNotSentDelayed':0}
    METASTATS['COMPSTATS'] = {'PADistCalc':0,'PAAffCalc':0,'R2Calcs':0,'MICalcs':0,'PerfComp':0,'Clustering':0}

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    s.sendall(ID.encode())
    print("connection established")
    print(ID, PORT, SIM_FROM, SIM_TO, FP, INIT_DAYS, STABLE_SIZE, MAX_WINDOW, ADWIN_DELTA)
    print("stablesixe: "+str(STABLE_SIZE))
    print(" maxwindow: "+str(MAX_WINDOW))
    print("adwin_delta: "+str(ADWIN_DELTA))

    ack = s.recv(1024).decode()
    print(repr(ack))
    simTarget(WEIGHTTYPE,s)

    print("---TRANSMODELSTATS----")
    print(METASTATS['TRANSMODELSTATS'])
    print("---LOCALMODELSTATS----")
    print(METASTATS['LOCALMODELSTATS'])
    print("---REPLACEMENTMODELSTATS----")
    print(METASTATS['REPLACEMENTMODELSTATS'])
    print("---COMMSTATS----")
    print(METASTATS['COMMSTATS'])
    print("---COMPSTATS----")
    print(METASTATS['COMPSTATS'])
    s.close()


if __name__ == '__main__':main()

