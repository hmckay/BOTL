import sys
import numpy as np
import pandas as pd
import time
import socket
import pickle
import time
import os.path
import math
from numpy.linalg import svd as SVD
from optparse import OptionParser
from scipy.spatial.distance import euclidean as euclid
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.kernel_ridge import KernelRidge as KRidge
from sklearn.linear_model import ElasticNet as Elastic
from sklearn.feature_selection import mutual_info_regression

# from Models import createModel,modelHistory
# from Models import modelMultiConceptTransfer as modelMulti
from Models import modelMultiConceptTransferHistoryRePro as modelMultiHistoryRepro
import preprocessData as preprocess
from datetime import datetime,timedelta
from sklearn import metrics
from scipy.stats import pearsonr

from scipy.linalg import norm
from sklearn.metrics.pairwise import polynomial_kernel as kernel
from Models.stsc_ulti import affinity_to_lap_to_eig, reformat_result, get_min_max
from Models.stsc_np import get_rotation_matrix as get_rotation_matrix_np

META_SIZE = 10
def calcError(y,preds):
    return metrics.r2_score(y,preds)

def receivePickleObject(s,numPackets,lenofObject):
    # ID = msg.split(',')[1]
    # numPackets = int(msg.split(',')[2])
    # lenofObject = int(msg.split(',')[3])
    print("NUMBER OF PACKETS EXPECTING: "+str(numPackets))
    s.sendall(('ACK,'+str(numPackets)).encode())
    
    pickledObject = b''
    while (len(pickledObject) < lenofObject):
        pickledObject = pickledObject+s.recv(1024)
    obj = pickle.loads(pickledObject)
    return obj

def sendPickleObject(modelID,brokenPickle,s):
    for i in brokenPickle:
        s.sendall(i)
    recACK = s.recv(1024).decode()
    print(recACK)
    if str(modelID) == str(recACK.split(',')[1]):
        return 1
    return 0

def replacementModelTransfer(sourceID,modelID,subModelID,substituteModelInfo,s,modelsSent,neighbors,METASTATS):
    replacementFlag = 0
    METASTATS['REPLACEMENTMODELSTATS']['modelsReplaced'] +=1
    METASTATS['REPLACEMENTMODELSTATS']['replacedModelIDs'].append(uniqueModID(modelID,sourceID))
    METASTATS['REPLACEMENTMODELSTATS']['replacementModelIDs'].append(uniqueModID(subModelID,sourceID))
    if sourceID in uniqueModID(subModelID,sourceID):
        METASTATS['REPLACEMENTMODELSTATS']['replacedWithLocal']+=1
    else:
        METASTATS['REPLACEMENTMODELSTATS']['replacedWithTrans']+=1
    replaceAndSendList = getReplacementLists(uniqueModID(modelID,sourceID),uniqueModID(subModelID,sourceID),s)
    modelToSend = pickle.dumps(substituteModelInfo)
    lenofModel = len(modelToSend)
    brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
    numPackets = len(brokenBytes)
    METASTATS['COMMSTATS']['ReplaceList']+=len(neighbors)

    if len(replaceAndSendList)>0:
        ReplaceModelmsg = 'ReplaceModel,True,'+str(uniqueModID(subModelID,sourceID))+','+str(numPackets)+','+str(lenofModel)
        METASTATS['COMMSTATS']['NumUniqueModelsSent']+=1
        METASTATS['COMMSTATS']['NumUniqueModelsSentBytes']+=lenofModel
        METASTATS['COMMSTATS']['NumModelsSent']+=len(replaceAndSendList)
        METASTATS['COMMSTATS']['NumModelsSentBytes']+=lenofModel*len(replaceAndSendList)
        s.sendall(ReplaceModelmsg.encode())
        replaceModelACK = s.recv(1024).decode()
        if int(replaceModelACK.split(',')[1])==numPackets:
            replacementFlag = sendPickleObject(uniqueModID(subModelID,sourceID),brokenBytes,s)
            print("sent replacement model: "+str(replacementFlag)+' '+str(subModelID))
            if not (uniqueModID(subModelID,sourceID) in modelsSent):
                modelsSent.append(uniqueModID(subModelID,sourceID))
    else:
        ReplaceModelmsg = 'ReplaceModel,False'
        s.sendall(ReplaceModelmsg.encode())
        ack = s.recv(1024).decode()
        if str(ack.split(',')[0])=='ACK':
            replacementFlag = 1

    return replacementFlag,modelsSent,METASTATS

def getReplacementLists(modelID,subModelID,s):
    REPLACEmsg = 'REPLACE,'+str(modelID)+','+str(subModelID)
    s.sendall(REPLACEmsg.encode())
    replaceListDetails = s.recv(1024).decode()
    if str(replaceListDetails.split(',')[0])=='ReplaceList':
        numPackets = int(replaceListDetails.split(',')[2])
        lenofReplaceList = int(replaceListDetails.split(',')[3])
        print("got ReplaceList header: "+str(replaceListDetails))

        replaceListToSend = receivePickleObject(s,numPackets,lenofReplaceList)
        # s.sendall(('ACK,'+str(numPackets)).encode())
        # replaceListToSend = receiveReplaceList(lenofReplaceList,s)
        print("replace list: "+str(replaceListToSend))
        return replaceListToSend
    return []


# def receiveReplaceList(lenofListPickle,s):
    # pickledReplaceList = b''
    # while (len(pickledReplaceList)<lenofListPickle):
        # pickledReplaceList = pickledReplaceList + s.recv(1024)
    # # conn.sendall(('RECEIVED,'+str(modelID)).encode())
    # replaceModelReceivers = pickle.loads(pickledReplaceList)
    # return replaceModelReceivers

    # if len(replaceModelReceivers)>0:

    # s.sendall(('ACK,'+str(numPackets)).encode())
def uniqueModID(modelID,sourceID):
    if '-' not in str(modelID):
        modelID = str(sourceID)+'-'+str(modelID)
    return modelID

def delayedModelTransfer(sourceID,modelID,model,s,modelsSent,delayedTransferList,METASTATS):
    tempModelsSent = []
    successFlag,tempModelsSent,METASTATS = modelReadyToSend(uniqueModID(modelID,sourceID),model,s,tempModelsSent,delayedTransferList,METASTATS)
    if successFlag and uniqueModID(modelID,sourceID) not in modelsSent:
        modelsSent.append(uniqueModID(modelID,sourceID))
    return successFlag,modelsSent,METASTATS

def checkSimilarModelTransfer(sourceID,modelID,model,conceptSimGroup,s,modelsSent,METASTATS):
    if uniqueModID(modelID,sourceID) not in modelsSent:
        conceptSimModList = []
        for c in conceptSimGroup:
            if '-' not in str(c):
                conceptSimModList.append(str(sourceID)+'-'+str(c))
            else:
                conceptSimModList.append(c)
        # modelID = uniqueModID(modelID,sourceID)
        noSimModelList,delayedSendList,METASTATS = CSMhandshake(uniqueModID(modelID,sourceID),conceptSimModList,s,METASTATS)
        if noSimModelList == 1 and delayedSendList == 1:
            modelsSent.append(uniqueModID(modelID,sourceID))
            print("noSimModelList and delayedSend are 1")
            return 1,modelsSent,[],METASTATS
        # if len(delayedSendList)>0:
            # METASTATS['LOCALMODELSTATS']['modelsDelayed']+=1
            # METASTATS['LOCALMODELSTATS']['modelsDelayedID'].append(uniqueModID(modelID,sourceID))
        if len(noSimModelList)==0:
            DELAYmsg = 'DELAY,'+str(uniqueModID(modelID,sourceID))
            s.sendall(DELAYmsg.encode())
            recACK = s.recv(1024).decode()
            return 1,modelsSent,delayedSendList,METASTATS

        successFlag,modelsSent,METASTATS = modelReadyToSend(uniqueModID(modelID,sourceID),model,s,modelsSent,noSimModelList,METASTATS)
        return successFlag,modelsSent,delayedSendList,METASTATS
    else:
        return 0,0,0,METASTATS
#send: CSM,modID,numPackets,lenPickle
#rec: ACK,numPacketsExpecting
def CSMhandshake(modelID,conceptSimGroup,s,METASTATS):
    listOfSimModels=pickle.dumps(conceptSimGroup)
    lenListOfSimModelPickle = len(listOfSimModels)
    brokenBytesSimModelList = [listOfSimModels[i:i+1024] for i in range(0,len(listOfSimModels),1024)]
    numPackets = len(brokenBytesSimModelList)
    CSMmsg = 'CSM,'+str(modelID)+','+str(numPackets)+','+str(lenListOfSimModelPickle)
    print("CSM message:"+str(CSMmsg))
    print("conceptSimgroup")
    print(conceptSimGroup)
    s.sendall(CSMmsg.encode())
    ack = s.recv(1024).decode()
    print(ack)
    if 'NOTNEEDED' in str(ack.split(',')[0]):
        return 1,1,METASTATS

    ackNumPackets = int(ack.split(',')[1])

    if ackNumPackets == numPackets:
        METASTATS['COMMSTATS']['CSMList']+=1
        METASTATS['COMMSTATS']['CSMListBytes']+=lenListOfSimModelPickle
        return sendSimModelList(modelID,brokenBytesSimModelList,s,METASTATS)
    print("exiting on CSMhandshake")    
    return 0,0,METASTATS
#send: pickle split into 1024 bytes
#rec: ACK,modelID,CSM or ACK,modelID,None
def sendSimModelList(modelID,brokenBytesModelList,s,METASTATS):
    print(len(brokenBytesModelList))
    for i in brokenBytesModelList:
        s.sendall(i)
        print("finished sending")
    # recACK = s.recv(1024).decode()
    # print(recACK)
    # if recACK.split(',')[1] != str(modelID):
        # return 0
    recACK = s.recv(1024).decode()
    print("recACK:"+ str(recACK))
    if str(recACK.split(',')[0]) == 'CSMdict':
        mID = recACK.split(',')[1]
        numPackets = int(recACK.split(',')[2])
        lenofCSMdict = int(recACK.split(',')[3])
        print("got CSMdict header: "+str(recACK))

        # s.sendall(('ACK,'+str(numPackets)).encode())

        return receiveCSMDict(modelID,numPackets,lenofCSMdict,s,METASTATS)
    print("existing on sendSimModelList")    
    return 0,0,METASTATS

def receiveCSMDict(modelID,numPackets,lenofCSMdict,s,METASTATS):
    # pickledCSMDict = b''
    # while (len(pickledCSMDict)<lenofCSMdict):
        # pickledCSMDict = pickledCSMDict + s.recv(1024)
    # # conn.sendall(('RECEIVED,'+str(modelID)).encode())
    # CSMDict = pickle.loads(pickledCSMDict)
    CSMDict = receivePickleObject(s,numPackets,lenofCSMdict)
    s.sendall(('ACK,'+str(numPackets)).encode())
    ack = s.recv(1024).decode()
    if str(ack.split(',')[0]) != 'CSMDetailsComplete':
        return []
    print("gotCSMDict: "+str(CSMDict))
    noSimilarModels = []
    delayedTransferList = []
    for n in CSMDict.keys():
        if not CSMDict[n]:
            noSimilarModels.append(n)
        else:
            delayedTransferList.append(n)
    # if noSimilarModels:
    print(noSimilarModels)
    return noSimilarModels,delayedTransferList,METASTATS




def modelReadyToSend(modelID,model,s,modelsSent,noSimModelList,METASTATS):
    successFlag = 0
    if modelID not in modelsSent:
        successFlag,METASTATS = handshake(modelID,model,s,modelsSent,noSimModelList,METASTATS)
    else:
        print("model already sent")
        return 1,modelsSent,METASTATS

    if successFlag:
        modelsSent.append(modelID)
        METASTATS['LOCALMODELSTATS']['modelsTransferred']+=1
        METASTATS['LOCALMODELSTATS']['transferredModelIDs'].append(modelID)
        print("sucessfully sent model")
        return 1, modelsSent,METASTATS
    else:
        print("unsucessful send")
        return 0, modelsSent,METASTATS

def handshake(modelID,model,s,modelsSent,noSimModelList,METASTATS):
    print("in handshake")
    print(modelID)
    modelToSend = pickle.dumps(model)
    lenofModel = len(modelToSend)
    brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
    numPackets = len(brokenBytes)
    pickledListReceivers = pickle.dumps(noSimModelList)
    brokenListReceivers = [pickledListReceivers[i:i+1024] for i in range(0,len(pickledListReceivers),1024)]
    numPacketsReceiverList = len(brokenListReceivers)
    RTSmsg = 'RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel)+','+str(len(pickledListReceivers))+','+str(numPacketsReceiverList)
    s.sendall(RTSmsg.encode())
    print("rts message")
    print(RTSmsg)
    ack = s.recv(1024).decode()
    print("about to send list of receivers")
    print(ack)
    receiverListSent = sendPickleObject(modelID,brokenListReceivers,s)
    ackNumPackets = int(ack.split(',')[1])
    METASTATS['COMMSTATS']['NumUniqueModelsSent']+=1
    METASTATS['COMMSTATS']['NumUniqueModelsSentBytes']+=lenofModel
    METASTATS['COMMSTATS']['NumModelsSent']+=len(noSimModelList)
    METASTATS['COMMSTATS']['NumModelsSentBytes']+=lenofModel*len(noSimModelList)

    if ackNumPackets == numPackets and receiverListSent:
        successFlag = sendPickleObject(modelID,brokenBytes,s)
        return successFlag,METASTATS#sendPickleObject(modelID,brokenBytes,s)
    print("failed in ihandshakes: "+str(modelID))
    return 0,METASTATS



# def sendModel(modelID,brokenBytes,s):
    # for i in brokenBytes:
        # s.sendall(i)
        # print("finished sending")
    # recACK = s.recv(1024).decode()
    # if str(modelID) == str(recACK.split(',')[1]):
        # return 1
    # return 0
def updateDiscardSourceModels(sourceModels,toDiscardDict,ID,METASTATS):
    print("need to DISCARD MODELS")
    print(toDiscardDict)
    print("source models:" +str(sourceModels.keys()))
    for mID in toDiscardDict.keys():
        METASTATS['COMMSTATS']['DiscardList']+=1
        subID = toDiscardDict[mID]['subModID']
        print("subID:"+str(subID))
        print("ID: "+str(ID))
        print("mid:"+str(mID))
        if mID in sourceModels.keys():
            sourceModels[mID]['toDiscard']=True
            if (ID in subID or subID in sourceModels):
                # sourceModels[mID]['toDiscard']=True
                METASTATS['TRANSMODELSTATS']['modelsDiscarded'] +=1
                METASTATS['TRANSMODELSTATS']['discardedModelIDs'].append(mID)
            else:
                print("submodel already been discarded")
                return 1, sourceModels,METASTATS
        else:
            print("already discarded "+str(ID))
            return 1, sourceModels,METASTATS
        # if mID in sourceModels.keys() and (ID in subID or subID in sourceModels):
                # sourceModels[mID]['toDiscard']=True
            # # else:
                # # print("dont have sub model available")
                # # return 0, sourceModels
        # else:
            # print("already discarded "+str(ID))
            # # return 1, sourceModels
            # return 0, sourceModels
    print(sourceModels)
    return 1, sourceModels,METASTATS



def readyToReceive(s,sourceModels,ID,METASTATS):
    s.sendall(('RTR,'+str(ID)).encode())
    print("sent RTR")
    data = s.recv(1024).decode()
    print("TARGET: num models to receive "+repr(data))
    ACKFlag = data.split(',')[0]
    numModels = int(data.split(',')[1])
    print("ready to recieve function called")
    if ACKFlag == 'ACK':
        if numModels == 0:
            print("nothing to receive")
            s.sendall(('END').encode())
            # return sourceModels,METASTATS
        else:
            s.sendall(('ACK').encode())
            for i in range(0,numModels):
                sourceModels,METASTATS = receiveModels(s,sourceModels,METASTATS)
                print("numModels: "+str(numModels))
                print("received: "+str(i))
    data = s.recv(1024).decode()
    if str(data.split(',')[0])=='NODISCARD':
        s.sendall(('NODISCARD,'+str(ID)).encode())
        end = s.recv(1024).decode()
        return sourceModels,METASTATS
    elif str(data.split(',')[0])=='DISCARD':
        numPackets = int(data.split(',')[2])
        lenofDict = int(data.split(',')[3])
        print("DISCARD MODEL NEEDED")
        print(data)
        print(numPackets,lenofDict)
        METASTATS['COMMSTATS']['DiscardListBytes']+=lenofDict
        toDiscardDict = receivePickleObject(s,numPackets,lenofDict)
        successFlag,sourceModels,METASTATS = updateDiscardSourceModels(sourceModels,toDiscardDict,ID,METASTATS)
        if successFlag:
            s.sendall(("DISCARDED,"+str(ID)).encode())
        
    return sourceModels,METASTATS


# def readyToReceive(s,sourceModels,ID):
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



def receiveModels(s,sourceModels,METASTATS):
    RTSInfo = s.recv(1024).decode()
    # RTSFlag = RTSInfo.split(',')[0]
    sourceModID = RTSInfo.split(',')[1]
    numPackets = int(RTSInfo.split(',')[2])
    lenofModel = int(RTSInfo.split(',')[3])
    modelInfo = receivePickleObject(s,numPackets,lenofModel)
    METASTATS['TRANSMODELSTATS']['modelsReceived']+=1
    METASTATS['TRANSMODELSTATS']['receivedModelIDs'].append(sourceModID)
    METASTATS['COMMSTATS']['NumModelsRec']+=1
    METASTATS['COMMSTATS']['NumModelsRecBytes']+=lenofModel
    # print("NUMBER OF PACKETS EXPECTING: "+str(numPackets))
    # s.sendall(('ACK,'+str(numPackets)).encode())
    
    # pickledModel = b''
    # while (len(pickledModel) < lenofModel):
        # pickledModel = pickledModel+s.recv(1024)
    s.sendall(('ACK,'+str(sourceModID)).encode())

    return storeSourceModel(sourceModID,modelInfo,sourceModels,METASTATS)
    # return storeSourceModel(sourceModID,pickledModel,sourceModels,METASTATS)
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

# def storeSourceModel(sourceModID,pickledModel,sourceModels,METASTATS):
def storeSourceModel(sourceModID,modelInfo,sourceModels,METASTATS):
    # print("picked model len is: "+str(len(pickledModel)))
    # model = pickle.loads(pickledModel)
    sourceModels[sourceModID] = modelInfo
    sourceModels[sourceModID]['toDiscard']=False
    METASTATS['modelsReceived'].append(sourceModID)
    # print(modelInfo['model'])
    print("storing source model: "+str(sourceModID)) 
    return sourceModels,METASTATS
# def storeSourceModel(sourceModID,pickledModel,sourceModels):
    # print("picked model len is: "+str(len(pickledModel)))
    # model = pickle.loads(pickledModel)
    # sourceModels[sourceModID] = model
    # print(model['model'])
    # print("storing source model: "+str(sourceModID)) 
    # return sourceModels


def updatePADistanceMatrix(newID,otherModels,dM,distanceMetric,METASTATS):
    if dM is None: dM = dict()
    dM[newID]=dict()
    totalCalcs = 0
    for j in dM.keys():
        totalCalcs +=1
        distance = distanceMetric(otherModels[newID]['PCs'],otherModels[j]['PCs'])
        METASTATS['COMPSTATS']['PADistCalc']+=1
        dM[newID][j] = distance
        dM[j][newID] = distance
    # print("distance matrix is: ")
    # print(dM)
    return dM,totalCalcs,METASTATS

def getSimMatrix(affDict,simKeys):
    simMatrix = np.zeros((len(simKeys),len(simKeys)))

    for idx,i in enumerate(simKeys):
        similarity = np.zeros(len(simKeys))
        for jdx,j in enumerate(simKeys):
            similarity[jdx]=affDict[i][j]
        simMatrix[idx]=similarity

    return simMatrix

def updateEuclidDistanceMatrix(newID,otherModels,dM,X,DROP_FIELDS,METASTATS,tLabel):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    dM[newID]=dict()

    for j in dM.keys():
        distance = euclid(otherModels[newID]['model'].predict(X),otherModels[j]['model'].predict(X))
        METASTATS['COMPSTATS']['PADistCalc']+=1
        dM[newID][j] = distance
        dM[j][newID] = distance
    # print("distance matrix is: ")
    # print(dM)
    return dM,METASTATS

def self_tuning_spectral_clustering(affinity, modelKeys, get_rotation_matrix, min_n_cluster=None, max_n_cluster=None):
    
    w, v = affinity_to_lap_to_eig(affinity)
    min_n_cluster, max_n_cluster = get_min_max(w, min_n_cluster, max_n_cluster)
    if max_n_cluster > 10:
        max_n_cluster = 10
    re = []
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r = get_rotation_matrix(x, c)
        re.append((cost, x.dot(r)))
        print('n_cluster: %d \t cost: %f' % (c, cost))
    COST, Z = sorted(re, key=lambda x: x[0])[0]
    return reformat_result(np.argmax(Z, axis=1), Z.shape[0],modelKeys)


def self_tuning_spectral_clustering_np(affinity, names, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, names, get_rotation_matrix_np, min_n_cluster, max_n_cluster)

def getCluster(modelID,dM,METASTATS):
    modelKeys = list(dM.keys())
    affinityMatrix,METASTATS = getAffinityMatrix(dM,METASTATS)
    similarity_matrix = getSimMatrix(affinityMatrix,modelKeys)
    minClusters = None
    # groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix,similarity_matrix2, names, get_rotation_matrix_np, minClusters, None)
    # print(similarity_matrix)
    groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix, modelKeys, get_rotation_matrix_np, minClusters, None)
    METASTATS['COMPSTATS']['Clustering']+=1
    print(groupedName)
    print(groupedID)
    
    for c in groupedName:
        if modelID in c:
            return c,METASTATS
    else:
        return groupedName[0],METASTATS


def getPerfCluster():
    pass

def evalConceptSimilar(currentModelID, conceptSimPreds,tLabel):
    print("evaluating conceptually similar")
    print(conceptSimPreds)
    possibleModels = list(conceptSimPreds.columns)
    possibleModels.remove(tLabel)
    print("possible models: "+str(possibleModels))
    print(len(conceptSimPreds))
    modelPerfs = dict()
    for i in possibleModels:
        try:
            modelPerfs[i] = calcError(conceptSimPreds[tLabel],conceptSimPreds[i])
        except:
            print(conceptSimPreds)
            print(possibleModels)
            exit()

    print(modelPerfs)
    maxModelID = max(modelPerfs,key=modelPerfs.get)
    return maxModelID
    
def getMISimModels(modelID,targetModel,window,modelSet,METASTATS,DROP_FIELDS,tLabel,stableLocals=None):
    totalCalcs = 0
    simModelList = []
    X = window.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = window[tLabel].copy()
    # predX = pd.DataFrame(columns = list(existingModels.keys()))
    predA = targetModel.predict(X)
    print(modelSet)
    print(stableLocals)
    if modelSet is not None:
        print("checkingMI:"+str(modelSet.keys()))
        for k,v in modelSet.items():
            if ('toDiscard' in v.keys() and v['toDiscard'] == False) or ('substituted' in v.keys() and v['substituted']==False) :
                predB = modelSet[k]['model'].predict(X)
                predDFs=pd.DataFrame(dict(predA=predA,predB=predB))
                mi = mutual_info_regression(predDFs,predB)
                METASTATS['COMPSTATS']['MICalcs']+=1
                mi=mi[0]/mi[1]
                totalCalcs +=1
                print("model "+str(k)+": "+str(mi))
                # if len(existingModels)>20:
                    # print(str(k)+": "+str(mi))
                if mi > 0.2 and k not in simModelList:
                    simModelList.append(k)
    # if len(existingModels)>20:
        # print(existingModels)
        # exit()
    if stableLocals is not None:
        print("checkingMI:"+str(stableLocals.keys()))
        for k,v in stableLocals.items():
            if v['substituted']==False:
                print("checkingMI:"+str(k))
                predB = stableLocals[k]['model'].predict(X)
                predDFs=pd.DataFrame(dict(predA=predA,predB=predB))
                mi = mutual_info_regression(predDFs,predB)
                METASTATS['COMPSTATS']['MICalcs']+=1
                mi=mi[0]/mi[1]
                totalCalcs +=1
                # if len(existingModels)>20:
                    # print(str(k)+": "+str(mi))
                # if mi > 0.2:
                if mi > 0.2 and k not in simModelList:
                    simModelList.append(k)

    return simModelList,totalCalcs,METASTATS


def checkToSendMI(modelID,model,window,existingModels,modelsSent,weightType,DROP_FIELDS,METASTATS,tLabel,UID):
    totalCalcs = 0
    if len(modelsSent)<2:
        return True,totalCalcs,METASTATS
    miSimModels,totalCalcs,METASTATS = getMISimModels(modelID,model,window,existingModels,METASTATS,DROP_FIELDS,tLabel)
    countModelsSent =len([v for v in miSimModels if uniqueModID(v,UID) in modelsSent])
    print("models sent that are in the same cluster")
    print(miSimModels)
    print(countModelsSent)
    print(modelsSent)
    if countModelsSent > 1:
        return False,totalCalcs,METASTATS 

    return True,totalCalcs,METASTATS 



def checkToSend(modelID,window,existingModels,modelsSent,localDistanceMatrix,weightType,DROP_FIELDS,METASTATS,tLabel,UID):
    totalCalcs = 0
    if len(modelsSent) < 2:
        localDistanceMatrix,tc,METASTATS = updatePADistanceMatrix(modelID,existingModels,localDistanceMatrix,principalAngles,METASTATS)
        return True,localDistanceMatrix,tc,METASTATS
    if 'PA' in weightType:
        localDistanceMatrix,tc,METASTATS = updatePADistanceMatrix(modelID,existingModels,localDistanceMatrix,principalAngles,METASTATS)
        totalCalcs +=tc
    else:
        localDistanceMatrix,tc,METASTATS = updateEuclidDistanceMatrix(modelID,existingModels,localDistanceMatrix,window,DROP_FIELDS,METASTATS,tLabel)
        totalCalcs +=tc
    clusterNewModel,METASTATS = getCluster(modelID,localDistanceMatrix,METASTATS)
    countModelsSent =len([v for v in clusterNewModel if uniqueModID(v,UID) in modelsSent])
    print("models sent that are in the same cluster")
    print(clusterNewModel)
    print(countModelsSent)
    print(modelsSent)
    if countModelsSent > 1:
        return False,localDistanceMatrix,totalCalcs,METASTATS

    return True,localDistanceMatrix,totalCalcs,METASTATS


def updateModelSetDM(modelID,substituteModelID,modelSet,existingModels,distanceMatrix):
    print("modelset at update: "+str(modelSet.keys()))
    print("subing model "+str(modelID)+" for "+str(substituteModelID))
    if modelID in modelSet.keys() and existingModels[modelID]['substituted']==True:
        del modelSet[modelID]
    if substituteModelID not in modelSet.keys():
        modelSet[substituteModelID]=existingModels[substituteModelID]
    if distanceMatrix and modelID in distanceMatrix.keys():
        del distanceMatrix[modelID]
        for k in distanceMatrix.keys():
            del distanceMatrix[k][modelID]
    print("modelset after update: "+str(modelSet.keys()))
    return modelSet,distanceMatrix


def removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix):
    subedSourceModels = []
    if sourceModels:
      for mID in sourceModels.keys():
            if sourceModels[mID]['toDiscard']==True:
                subedSourceModels.append(mID)
    subedExistingModels = []
    if existingModels:
        for mID in existingModels.keys():
            if existingModels[mID]['substituted']==True:
                subedExistingModels = []
    if subedSourceModels:
        print('substituted models that need discarding are:')
        print(subedSourceModels)
        print("original")
        print(modelSet.keys())
        print(sourceModels.keys())
    if subedExistingModels:
        print('substituted models that need discarding are:')
        print(subedExistingModels)
        print("original")
        print(modelSet.keys())
        print(existingModels.keys())

    for mID in subedExistingModels:
        if existingModel is not None and mID in existingModels.keys():
            del existingModels[mID]
        if modelSet is not None and mID in modelSet.keys():
            del modelSet[mID]
        if distanceMatrix is not None and mID in distanceMatrix.keys():
            del distanceMatrix[mID]
            for k in distanceMatrix.keys():
                del distanceMatrix[k][mID]
    for mID in subedSourceModels:
        if sourceModels is not None and mID in sourceModels.keys():
            del sourceModels[mID]
        if modelSet is not None and mID in modelSet.keys():
            del modelSet[mID]
        if distanceMatrix is not None and mID in distanceMatrix.keys():
            del distanceMatrix[mID]
            for k in distanceMatrix.keys():
                del distanceMatrix[k][mID]
    if subedSourceModels:
        print('AFTER substituted models that need discarding are:')
        print(subedSourceModels)
        print("original")
        print(modelSet.keys())
        print(sourceModels.keys())
    if subedExistingModels:
        print('AFTER substituted models that need discarding are:')
        print(subedExistingModels)
        print("original")
        print(modelSet.keys())
        print(existingModels.keys())
    return existingModels,sourceModels,modelSet,distanceMatrix


def addToModelSet(lastModID,existingModels,sourceModels,modelSet,WEIGHTTYPE,PATHRESH,METASTATS,stableThreshold,distanceMatrix=None,REPLACEMENT=False):
    existingModels,sourceModels,modelSet,distanceMatrix = removeDiscardedModels(existingModels,sourceModels,modelSet,distanceMatrix)
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    uniqueModel = True
    # print("adding new model to model set: "+str(lastModID))
    # print(existingModels)
    totalCalcs = 0
    for newID in newSources:
        if WEIGHTTYPE == 'OLSFEPA':
            uniqueModel,sourceModels,distanceMatrix,tc,METASTATS = checkAngles(newID,sourceModels,distanceMatrix,PATHRESH,METASTATS)
            totalCalcs +=tc
        if uniqueModel:
            modelSet[newID] = sourceModels[newID]
            modelSet[newID]['delAWE']=0
    if 'PAC' in WEIGHTTYPE or 'CL' in WEIGHTTYPE:# or REPLACEMENT == True:#'replace' in WEIGHTTYPE:
        if lastModID is not None and lastModID not in modelSet.keys() and 'temp' not in str(lastModID):
            if modelMultiHistoryRepro.isStable(lastModID,existingModels,stableThreshold) and existingModels[lastModID]['substituted']==False:
                modelSet[lastModID] = existingModels[lastModID]
                modelSet[lastModID]['delAWE']=0
    return modelSet,sourceModels,distanceMatrix,totalCalcs,METASTATS

def AWEaddToModelSet(df,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS,METASTATS,stableThreshold):
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    for newID in newSources:
        numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
        modelSet[newID] = sourceModels[newID]
        # if numAWEModels<=META_SIZE-1:
        modelSet[newID]['delAWE'] = 0
        # else:
        while numAWEModels>META_SIZE-1:
            modelSet = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,sourceModels[newID]['model'],newID,tLabel,DROP_FIELDS)
            METASTATS['COMPSTATS']['PerfComp']+=len(modelSet)
            numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
    if lastModID is not None:
        if modelMultiHistoryRepro.isStable(lastModID,existingModels,stableThreshold):
            numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
            modelSet[lastModID] = existingModels[lastModID]
            # if numAWEModels<=META_SIZE-1:
            modelSet[lastModID]['delAWE'] = 0
            # else:
            while numAWEModels>META_SIZE-1:
                modelSet,METASTATS = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,existingModels[lastModID]['model'],
                        lastModID,tLabel,DROP_FIELDS,METASTATS)
                METASTATS['COMPSTATS']['PerfComp']+=len(modelSet)
                numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
    return modelSet,METASTATS

def AddExpPaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,stableThreshold):
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    metaModel = weights['metaModel']
    for newID in newSources:
        if not metaModel:
            metaModel = dict()
        # metaModel[newID]=1
        numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
        modelSet[newID] = sourceModels[newID]
        modelSet[newID]['delAddExp'] = 0
        while numAddExpModels>META_SIZE-1:
            # modelKeys = list(set(metaModel.keys())& set(modelSet.keys())
            minW = min(metaModel.values())
            minWeightID = [k for k,v in metaModel.items() if v == minW][0]
            minWeightID = min(metaModel.keys() & modelSet.keys(), key = metaModel.get)
            print("metamodel items:"+str(metaModel.items()))
            print("modelSet:"+str(modelSet))
            modelSet[minWeightID]['delAddExp']=1
            numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
            del metaModel[minWeightID]
        modelSet[newID]['delAddExp'] = 0
        weights['metaModel'] = metaModel
        metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
    if lastModID is not None:
        if modelMultiHistoryRepro.isStable(lastModID,existingModels,stableThreshold):
            numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
            modelSet[lastModID] = existingModels[lastModID]
            modelSet[lastModID]['delAddExp'] = 0
            while numAddExpModels>META_SIZE-1:
                minW = min(metaModel.values())
                minWeightID = [k for k,v in metaModel.items() if v == minW][0]
                minWeightID = min(metaModel.keys() & modelSet.keys(), key = metaModel.get)
                modelSet[minWeightID]['delAddExp']=1
                numAddExpModels = sum([1 for i in modelSet.values() if i['delAddExp'] == 0])
                del metaModel[minWeightID]
            modelSet[lastModID]['delAddExp'] = 0
        else:
            del metaModel[lastModID]
    return modelSet,orderedModels,metaModel,METASTATS


def AddExpOaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS,METASTATS,stableThreshold):
    newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    metaModel = weights['metaModel']
    for newID in newSources:
        if not metaModel:
            metaModel = dict()
        # metaModel[newID]=1
        modelSet[newID] = sourceModels[newID]
        orderedModels.append(newID)
        while len(orderedModels)>META_SIZE-1:
            oldestID = orderedModels.pop(0)
            modelSet[oldestID]['delAddExp']=1
            del metaModel[oldestID]
        modelSet[newID]['delAddExp'] = 0
        weights['metaModel'] = metaModel
        metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
    if lastModID is not None:
        if modelMultiHistoryRepro.isStable(lastModID,existingModels,stableThreshold):
            modelSet[lastModID] = existingModels[lastModID]
            orderedModels.append(lastModID)
            if len(orderedModels)>META_SIZE-1:
                oldestID = orderedModels.pop(0)
                modelSet[oldestID]['delAddExp']=1
                del metaModel[oldestID]
            modelSet[lastModID]['delAddExp'] = 0
        else:
            del metaModel[lastModID]
    return modelSet,orderedModels,metaModel,METASTATS

def principalAngles(x,y):
    swapped = False
    if y.shape[1] < x.shape[1]:
        swapped=True
    yshape = y.shape[1]
    if y.shape[1] < x.shape[1]:
        yT = y.transpose()
        yTx = np.dot(yT,x)
        u,sig,v = SVD(yTx)
        angles = np.zeros(len(sig))
        yshape = x.shape[1]
    else:
        xT = x.transpose()
        xTy = np.dot(xT,y)
        u,sig,v = SVD(xTy)
        angles = np.zeros(len(sig))
    # print("sig:"+str(sig))
    # print("shape of x: "+str(x.shape[1]))
    # print("shape of y: "+str(y.shape[1]))
    # print("swapped: "+str(swapped))
    for idx,a in enumerate(sig):
        ang = a#np.round(a,decimals=14)
        # print(idx,a)
        if a>=1:
            angles[idx] = np.arccos(1)
        else:
            angles[idx] = np.arccos(ang)
    tot = 0
    for a in angles:
        tot += np.cos(a)/yshape
    return (1 - tot)

def checkAngles(newID,sourceModels,distanceMatrix,PAThresh,METASTATS):
    totalCalcs = 0
    if distanceMatrix is None: distanceMatrix = dict()
    unique = True
    
    distanceMatrix,tc,METASTATS = updatePADistanceMatrix(newID,sourceModels,distanceMatrix,principalAngles,METASTATS)
    totalCalcs+=tc

    # distanceMatrix[newID]=dict()
    # for j in distanceMatrix.keys():
        # distance = principalAngles(sourceModels[newID]['PCs'],sourceModels[j]['PCs'])
        # distanceMatrix[newID][j] = distance
        # distanceMatrix[j][newID] = distance
    # print("distance matrix is: ")
    # print(distanceMatrix)
    
    affinityMatrix,METASTATS = getAffinityMatrix(distanceMatrix,METASTATS)
    print("affinity matrix is: ")
    print(affinityMatrix)

    if len(affinityMatrix) <=1:
        return unique,sourceModels,distanceMatrix,totalCalcs,METASTATS
    closestAffinity = sorted(affinityMatrix[newID].values(),reverse=True)[1]
    
    if closestAffinity >= PAThresh:
        unique = False

        print("Removing from model set"+str(newID))
        del distanceMatrix[newID]
        for j in distanceMatrix.keys():
            del distanceMatrix[j][newID]
        del sourceModels[newID]
    return unique,sourceModels,distanceMatrix,totalCalcs,METASTATS


def getAffinityMatrix(distanceMatrix,METASTATS):
    k=3
    affinityMatrix = dict()
    for i in distanceMatrix.keys():
        affinityMatrix[i]=dict()
        iZeroNeighbours = sum(1 for vals in distanceMatrix[i].values() if vals==0)
        if k > iZeroNeighbours and k <len(distanceMatrix.keys()):
            kNN = k
        elif k <=iZeroNeighbours and iZeroNeighbours < len(distanceMatrix.keys()):
            kNN = iZeroNeighbours
        else:
            kNN = len(distanceMatrix.keys())-1
        iNormaliser = sorted(distanceMatrix[i].values(),reverse=False)[kNN]

        for j in distanceMatrix.keys():
            jZeroNeighbours = sum(1 for vals in distanceMatrix[j].values() if vals==0)
            if k > jZeroNeighbours and k <len(distanceMatrix.keys()):
                kNN = k
            elif k <=jZeroNeighbours and jZeroNeighbours < len(distanceMatrix.keys()):
                kNN = jZeroNeighbours
            else:
                kNN = len(distanceMatrix.keys())-1
            jNormaliser = sorted(distanceMatrix[j].values(),reverse=False)[kNN]
            
            if iNormaliser == 0 and jNormaliser == 0:
                iNormaliser = 1
                jNormaliser = 1
            elif iNormaliser == 0:
                iNormaliser = jNormaliser
            elif jNormaliser == 0:
                jNormaliser = iNormaliser

            affinityMatrix[i][j] =  math.exp(-(distanceMatrix[i][j]**2)/(iNormaliser*jNormaliser))
            METASTATS['COMPSTATS']['PAAffCalc']+=1

    return affinityMatrix,METASTATS

def genResultsHeader(headerSTR):

    headerSTR = headerSTR +('connDeg,R2,RMSE,Err,Pearsons,PearsonsPval,sourceModels,'+ 
            'reproR2,reproRMSE,reproErr,reproPearsons,reproPearsonsPval,'+
            'modelsLearnt,stableModelsLearnt,modelsTransferred,modelsDelayed,stableNeverSent,'+
            'modelsReplaced,replacedWithLocal,replacedWithTrans,'+
            'modelsReceived,modelsDiscarded,'+
            'numModelsSentP2P,CSMList,ReplaceList,DiscardList,'+
            'numUniqueModelsSentBytes,numModelsSentBytesP2P,numModelsReceivedBytesP2P,numModelsNotSentDelayed,'+
            'R2Calcs,MICalcs,PADistCalcs,PerfComp,triggerPACalcs,triggerMetricsCalc,Clustering,'+
            'numEvalModelSet,maxSizeModelSet,avgSizeModelSet,'+
            'meanNumModels,baseLearnerChange,avgBases,metaWeightUpdates,'+
            'neighbours,baseUsage\n')
            # 'conn_deg,neighbors,numModelsUsed,totalLocalModels,totalStableLocalModels,'+
            # 'METAnumModelsSent,METAstableLocals,METAnumModelsRec,METABaseChange,METABaseUsage,METAAvgBases,METAWeightUpdates,'+
            # 'totalPACalcs,totalOtherMetricCalcs,triggerPACalcs,triggerMetricCalcs\n')
    return headerSTR

def genResultsInstance(initResultsInstance,METASTATS):
    LOCAL = METASTATS['LOCALMODELSTATS']
    TRANS = METASTATS['TRANSMODELSTATS']
    REPLACE = METASTATS['REPLACEMENTMODELSTATS']
    COMM = METASTATS['COMMSTATS']
    COMP = METASTATS['COMPSTATS']
    
    #add local model results
    initResultsInstance = initResultsInstance + (str(LOCAL['modelsLearnt'])+','+str(LOCAL['stableModelsLearnt'])+','+
            str(LOCAL['modelsTransferred'])+','+str(LOCAL['modelsDelayed'])+','+str(LOCAL['stableNeverSent'])+',')
    #replacemnt model results
    initResultsInstance = initResultsInstance + (str(REPLACE['modelsReplaced'])+','+
            str(REPLACE['replacedWithLocal'])+','+str(REPLACE['replacedWithTrans'])+',')
    #add trans model results
    initResultsInstance = initResultsInstance + (str(TRANS['modelsReceived'])+','+str(TRANS['modelsDiscarded'])+',')
    #comm stats
    initResultsInstance = initResultsInstance + (str(COMM['NumModelsSent'])+','+str(COMM['CSMList'])+','+
        str(COMM['ReplaceList'])+','+str(COMM['DiscardList'])+','+str(COMM['NumUniqueModelsSentBytes'])+','+
        str(COMM['NumModelsSentBytes'])+','+str(COMM['NumModelsRecBytes'])+','+str(COMM['NumModelsNotSentDelayed'])+',')
    #comp stats
    initResultsInstance = initResultsInstance +(str(COMP['R2Calcs'])+','+str(COMP['MICalcs'])+','+
        str(COMP['PADistCalc'])+','+str(COMP['PerfComp'])+',')
    #others
    initResultsInstance = initResultsInstance +(str(METASTATS['triggerPACalcs'])+','+str(METASTATS['triggerMetricCalcs'])+','+
        str(COMP['Clustering'])+','+str(METASTATS['evalModelSet'])+',')
    #modelset stats
    # print(METASTATS['sizeModelSet'])
    if len(METASTATS['sizeModelSet'])==0:
        METASTATS['sizeModelSet']=[0]
    initResultsInstance = initResultsInstance + (str(max(METASTATS['sizeModelSet']))+','+
        str(sum(METASTATS['sizeModelSet'])/len(METASTATS['sizeModelSet']))+',')


    return initResultsInstance



# def AWEaddToModelSet(df,lastModID,existingModels,sourceModels,modelSet,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # for newID in newSources:
        # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
        # modelSet[newID] = sourceModels[newID]
        # if numAWEModels<META_SIZE-1:
            # modelSet[newID]['delAWE'] = 0
        # else:
            # modelSet = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,sourceModels[newID]['model'],newID,tLabel,DROP_FIELDS)
    # if lastModID is not None and lastModID not in modelSet.keys():
        # if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            # numAWEModels = sum([1 for i in modelSet.values() if i['delAWE'] == 0])
            # modelSet[lastModID] = existingModels[lastModID]
            # if numAWEModels<META_SIZE-1:
                # modelSet[lastModID]['delAWE'] = 0
            # else:
                # modelSet = modelMultiHistoryRepro.getBestAWEModels(df,modelSet,existingModels[lastModID]['model'],
                        # lastModID,tLabel,DROP_FIELDS)
    # return modelSet

# def AddExpPaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # metaModel = weights['metaModel']
    # print("modelset is")
    # print(modelSet)
    # for newID in newSources:
        # if not metaModel:
            # metaModel = dict()
        # # metaModel[newID]=1
        # modelSet[newID] = sourceModels[newID]
        # if len(metaModel.keys())>META_SIZE-1:
            # minW = min(metaModel.values())
            # minWeightID = [k for k,v in metaModel.items() if v == minW][0]
            # modelSet[minWeightID]['delAddExp']=1
            # del metaModel[minWeightID]
        # modelSet[newID]['delAddExp'] = 0
        # weights['metaModel'] = metaModel
        # metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
        # print("heremeta")
        # print(metaModel)
    # if lastModID is not None:# and lastModID not in modelSet.keys():
        # if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            # modelSet[lastModID] = existingModels[lastModID]
            # if len(metaModel.keys())>META_SIZE-1:
                # minW = min(metaModel.values())
                # minWeightID = [k for k,v in metaModel.items() if v == minW][0]
                # modelSet[minWeightID]['delAddExp']=1
                # del metaModel[minWeightID]
            # modelSet[lastModID]['delAddExp'] = 0
        # else:
            # del metaModel[lastModID]
    # return modelSet,orderedModels,metaModel


# def AddExpOaddToModelSet(df,idx,lastModID,existingModels,sourceModels,modelSet,orderedModels,weights,tLabel,DROP_FIELDS):
    # newSources = list(set(sourceModels.keys()) - set(modelSet.keys()))
    # newSources = [newID for newID in sourceModels.keys() if newID not in modelSet.keys()]
    # metaModel = weights['metaModel']
    # print("modelset is")
    # print(modelSet)
    # for newID in newSources:
        # if not metaModel:
            # metaModel = dict()
        # # metaModel[newID]=1
        # modelSet[newID] = sourceModels[newID]
        # orderedModels.append(newID)
        # if len(orderedModels)>META_SIZE-1:
            # oldestID = orderedModels.pop(0)
            # modelSet[oldestID]['delAddExp']=1
            # del metaModel[oldestID]
        # modelSet[newID]['delAddExp'] = 0
        # weights['metaModel'] = metaModel
        # metaModel = modelMulti.calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,modelSet,newID,modelSet[newID]['model'],weights)['metaModel']
        # print("heremeta")
        # print(metaModel)
    # if lastModID is not None:# and lastModID not in modelSet.keys():
        # if modelMultiHistoryRepro.isStable(lastModID,existingModels):
            # modelSet[lastModID] = existingModels[lastModID]
            # orderedModels.append(lastModID)
            # if len(orderedModels)>META_SIZE-1:
                # oldestID = orderedModels.pop(0)
                # modelSet[oldestID]['delAddExp']=1
                # del metaModel[oldestID]
            # modelSet[lastModID]['delAddExp'] = 0
        # else:
            # del metaModel[lastModID]
    # return modelSet,orderedModels,metaModel


