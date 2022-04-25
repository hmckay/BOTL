import numpy as np
import networkx as nx
import subprocess
import socket
import random
# import source
import pandas as pd
import pickle
import threading
import time
import sys
import os
from optparse import OptionParser
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.kernel_ridge import KernelRidge as KRidge
from sklearn.linear_model import ElasticNet as Elastic

MODELS = dict()
MODELSTORECEIVE = dict()
MODELSSENT = dict()
MODELSTOREPLACE = dict()
DISCARDEDMODELS = dict()
INIT_DAYS = 0#80
MODEL_HIST_THRESHOLD_PROB = 0# 0.4
MAX_WINDOW = 0#80
STABLE_SIZE = 0#2* MAX_WINDOW
MODEL_HIST_THRESHOLD_ACC = 0#0.5
THRESHOLD = 0#0.5
DEFAULT_PRED = ''
CD_TYPE = ''
NETWORK = None
NODE_ID_KEYS = dict()
REPLACEMENT = 0

class myThread (threading.Thread):
    def __init__(self,threadID,info,receivedModels,runnum,nums):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = info['Name']
        self.uid = info['uid']
        self.outputFile = info['stdo']
        self.PORT = info['PORT']
        self.fp = info['Run']
        self.inputFile = info['stdin']
        self.sFrom = info['sFrom']
        self.sTo = info['sTo']
        self.weightType = info['weightType']
        self.cullThresh = info['cullThresh']
        self.miThresh = info['miThresh']
        self.receivedModels = receivedModels
        self.runID = runnum
        self.numStreams = nums
        self.default_pred = info['default_pred']
        self.PCAVar = info['PCAVar']
        self.paThresh = info['paThresh']
        self.learnerType = info['learnerType']
        self.frameworkType = info['frameworkType']
        self.connDeg = info['connDeg']
        self.neighbors = info['neighbors']
        self.replace = info['replace']
    
    def run(self):
        print("starting "+ self.name)
        initiate(self.threadID,self.name,self.uid,self.PORT,self.learnerType,self.fp,self.inputFile,self.outputFile,self.
                sFrom,self.sTo,self.weightType,self.receivedModels,self.runID,self.numStreams,
                self.cullThresh,self.miThresh,self.paThresh,self.PCAVar,self.default_pred,self.frameworkType,self.connDeg,self.neighbors,self.replace)
        print("exiting " + self.name)

def sendPickledObject(sourceID,modelID,obj,conn,header):
    pickledObject = pickle.dumps(obj)
    lenPickle = len(pickledObject)
    brokenBytesPickle = [pickledObject[i:i+1024] for i in range(0,len(pickledObject),1024)]
    numPackets = len(brokenBytesPickle)
    msg = str(header)+str(modelID)+','+str(numPackets)+','+str(lenPickle)
    conn.sendall(msg.encode())
    # print(str(sourceID)+"S: "+str(msg))
    # print("sent pickleObject header")
    # print(msg)
    ack = conn.recv(1024).decode()
    # print(str(sourceID)+"R: "+str(ack))
    ackNumPackets = int(ack.split(',')[1])
    # print(ackNumPackets)
    # print(numPackets)
    # print(ack)
    # print("ack was above")

    if ackNumPackets == numPackets:
        # print("about to send ReplaceList")
        for i in brokenBytesPickle:
            conn.sendall(i)
        # print(str(sourceID)+"S: pickleEnd "+str(brokenBytesPickle[-1]))
            # print("finished sending")
        return 1
    return 0


def receivePickledObject(sourceID,modelID,lenReceiversPackets,conn,sendACK=True):
    pickledObject = b''
    while (len(pickledObject)<lenReceiversPackets):
        pickledObject = pickledObject + conn.recv(1024)

    # print(str(sourceID)+"R: pickledObj"+str(pickledObject[-8:])+"len: "+str(len(pickledObject))+"(of "+str(lenReceiversPackets)+")")
    if sendACK:
        conn.sendall(('RECEIVED,'+str(modelID)).encode())
        # print(str(sourceID)+"S: RECEIVED"+str(modelID))
    # print("sending receive pickle")
    # if 'RTR' in pickledObject[-8:]:
        # print(sourceID)
        # print(pickledObject)
    # if 'D003J005' in sourceID:
        # # print(pickledObject)
        # print(pickledObject[-5:])
    obj = pickle.loads(pickledObject)
    return obj

def getModelsToSend(threadID,modelsSent):
    toSend = dict()
    allModels = MODELS
    # print(threadID)
    # for ni in NETWORK.nodes():
        # print(ni)
        # print(list(NETWORK.neighbors(ni)))
    # print(threadID)
    # # print(list(NETWORK.neighbors(NODE_ID_KEYS[threadID])))
    # print(list(NETWORK.neighbors(threadID)))

    for mID,modeldict in MODELSTORECEIVE[threadID].items():
        if modeldict['sent']==False and modeldict['replace']==False:
            toSend[mID]=modeldict['model']
    # for tID,modelDict in allModels.items():
        # # if tID != threadID and NODE_ID_KEYS[tID] in list(NETWORK.neighbors(NODE_ID_KEYS[threadID])):
        # if tID != threadID and tID in list(NETWORK.neighbors(threadID)):
            # print("communication between: "+str(tID)+", "+str(threadID))
            # # print("aka: "+str(NODE_ID_KEYS[tID])+", "+str(NODE_ID_KEYS[threadID]))
            # for modelID,model in modelDict.items():
                # sourceModID = str(tID)+'-'+str(modelID)
                # print("sourceModID: "+str(sourceModID))
                # print("modelsSent: "+str(modelsSent))
                # if sourceModID not in modelsSent:
                    # toSend[sourceModID] = model
    return toSend

def getDictToDiscard(targetID):
    global MODELSTOREPLACE
    global MODELSTORECEIVE
    modelsToDiscard = dict()
    replacedBySelf = False
    replacedBySelfInfo = []
    for mID in MODELSTOREPLACE[targetID].keys():
        replacedBySelf=False
        if MODELSTOREPLACE[targetID][mID]['replaced']==False and (mID in MODELSTORECEIVE[targetID].keys() and MODELSTORECEIVE[targetID][mID]['sent']==True):
            subModID = MODELSTOREPLACE[targetID][mID]['subModelID']
            source = subModID.split('-')[0]
            if subModID not in MODELSTOREPLACE[targetID].keys() and subModID not in DISCARDEDMODELS[targetID]:
                modelsToDiscard[mID] = {'subModID':subModID,'source':source}#MODELSTORECEIVE[targetID][subModID]['source'],

    return modelsToDiscard


def checkAvailableReplacement(targetID):
    for mID in MODELSTOREPLACE[targetID].keys():
        subModelID = MODELSTOREPLACE[targetID][mID]['subModelID']
        source = subModelID.split('-')[0]
        if targetID not in source and (subModelID not in MODELSTORECEIVE[targetID].keys()):
            print("fail1 target :"+str(targetID)+" with model "+str(mID)+" replaced by: "+str(subModelID))
            return 0
        if targetID not in source and (MODELSTORECEIVE[targetID][subModelID]['sent']==False and MODELSTORECEIVE[targetID][subModelID]['replace']==False):
            print("fail2 target :"+str(targetID)+" with model "+str(mID)+" replaced by: "+str(subModelID))

            return 0
    return 1


def getDiscardAndReplace(targetID,modelsToReplace):
    modelsToDiscard = []
    modelsToReplaceSend = []

    for mID in modelsToReplace:
        if MODELSTOREPLACE[targetID][mID]['sendModel'] == False:
            modelsToDiscard.append(mID)
        else:
            modelsToReplaceSend.append(MODELSTOREPLACE[targetID][mID])
    return modelsToDiscard,modelsToReplaceSend

def updateReplaceDict(targetID,modelsToReplace):
    global MODELSTOREPLACE
    global MODELSTORECEIVE
    for mID in modelsToReplace:
        MODELSTOREPLACE[targetID][mID]['replaced']=True
        if mID in MODELSTORECEIVE[targetID].keys():
            MODELSTORECEIVE[targetID][mID]['replace']=True

def sendDiscardInfo(targetID, toDiscard,conn):
    print(str(targetID)+': is discarding '+str(toDiscard))
    successFlag = sendPickledObject(targetID,'None',toDiscard,conn,'DISCARD,')
    if successFlag:
        successFlag = 0
        recACK = conn.recv(1024).decode()
        # print(str(targetID)+"R: "+str(recACK))
        if str(recACK.split(',')[0])=='DISCARDED':
            successFlag = 1
        else:
            print(str(targetID)+" ack for dicard is: "+str(recACK))
        # print("recACK for Discard "+str(targetID)+":"+str(recACK))
    return successFlag



def sendHandshake(targetID,data,conn,modelsSent):
    RTRFlag = data.split(',')[0]
    # print(str(targetID)+"R: "+str(RTRFlag))
    successFlag = 0
    if RTRFlag == 'RTR':
        # target_ID = int(data.split(',')[1])
        target_ID = str(data.split(',')[1])
        # print("RTR ack: "+str(data))
        if target_ID != targetID:
            print("changed targetIDs")
            return 0,modelsSent
        toDiscard = getDictToDiscard(targetID)
        modelsToSend = getModelsToSend(targetID,modelsSent)
        numModels = len(modelsToSend)
        conn.sendall(('ACK,'+str(numModels)).encode())
        # print(str(targetID)+"S: ACK "+str(numModels))
        ack = conn.recv(1024).decode()
        # print(str(targetID)+"R: "+str(ack))
        # print(targetID, repr(ack))
        if ack == 'ACK':
            successFlag,modelsSent = sendModels(targetID,numModels,modelsToSend,modelsSent,conn)
            # print("ACK in sendHandshake" +str(successFlag))
        elif ack == 'END':
            successFlag = 1
        # toDiscard = getDictToDiscard(targetID)
        if len(toDiscard)>0 and successFlag:
            successFlag = sendDiscardInfo(targetID,toDiscard,conn)
            updateReplaceDict(targetID,toDiscard)
        else:
            conn.sendall(('NODISCARD,'+str(targetID)).encode())
            # print(str(targetID)+"S: NODISCARD"+str(targetID))
            # print("NODISCARD")
            ack = conn.recv(1024).decode()
            # print(str(targetID)+"R: "+str(ack))
            if str(ack.split(',')[0])=='NODISCARD':
                successFlag = 1
            if successFlag:
                conn.sendall(('END').encode())
                
                # print(str(targetID)+"S: END ")
                # print("NODISCARD ack")
                # print(str(ack))
            
#need to update MODELSTOREPLACE here
        return successFlag,modelsSent
    print("failed in sendHandshake: "+str(targetID))
    return 0,modelsSent

def sendModels(targetID,numModels,modelsToSend,modelsSent,conn):
    global MODELSTORECEIVE
    global MODELSSENT
    successFlag = 1
    for modelID,model in modelsToSend.items():
        print(str(targetID)+': is receiving model '+str(modelID))
        successFlag = sendPickledObject(targetID,modelID,model,conn,'RTS,')
        if successFlag:
            recACK = conn.recv(1024).decode()
            # print(str(targetID)+"R: "+str(recACK))
            # print("recACK"+str(recACK))
            if modelID == recACK.split(',')[1]:
                modelsSent.append(modelID)
                MODELSTORECEIVE[targetID][modelID]['sent']=True
                MODELSSENT[targetID].append(modelID)
            # print("models sent: "+str(modelsSent))
        else:
            print("!!!!!!!!!!!!!!!!!!!!!1"+str(targetID)+" failed to send model: "+str(modelID))
            return 0, modelsSent
    if successFlag:
        return 1, modelsSent
    print("success flag is 0 for "+str(targetID)+" failed to send model: "+str(modelID))
    return 0,modelsSent


def sendModel(modelID,brokenBytes,modelsSent,conn):
    for idx,i in enumerate(brokenBytes):
        conn.sendall(i)
    # print(str(sourceID)+"S: modelpickle"+str(brokenBytes[-1]))
    recACK = conn.recv(1024).decode()
    # print(str(sourceID)+"R: "+str(recACK))
    if modelID == recACK.split(',')[1]:
        modelsSent.append(modelID)
        # print("models sent: "+str(modelsSent))
        return 1, modelsSent
    return 0, modelsSent

def alreadyReceived(sourceID,modelID):
    neigh = list(NETWORK.neighbors(sourceID))
    allReceived = True
    if not REPLACEMENT:
        return False
    for n in neigh:
        if modelID in MODELSTORECEIVE[n].keys() or n in modelID:
            allReceived = True
        else:
            return False
    return allReceived


def checkSimModelsHandshake(sourceID,data,conn):
    CSMFlag = data.split(',')[0]
    # print(str(sourceID)+"R: "+str(data))
    if CSMFlag == 'CSM':
        modelID = data.split(',')[1]
        print(str(sourceID)+': wants to send model '+str(modelID))
        # print(modelID)
        numPackets = int(data.split(',')[2])
        lenofModelList = int(data.split(',')[3])
        if alreadyReceived(sourceID,modelID):
            conn.sendall(('NOTNEEDED,'+str(modelID)).encode())
            # print(str(sourceID)+"S: NOTNEEDED"+str(modelID))
            print(str(sourceID)+': doesnt need to send because all sources have it '+str(modelID))
            successFlag = 1
            return 1
        else:
            conn.sendall(('ACK,'+str(numPackets)).encode())
            # print(str(sourceID)+"S: ACK"+str(numPackets))

            successFlag = receiveCSMData(sourceID,modelID,numPackets,lenofModelList,conn)
        # print("CSM done - need RTS message:"+str(sourceID))

        rts = conn.recv(1024).decode()
        # print(str(sourceID)+"R: "+str(rts))
        if str(rts.split(',')[0])=='RTS':
            # print("RECEIVED RTS FROM: "+str(sourceID))
            # print(rts)
            successFlag = receiveHandshake(sourceID,rts,conn)
            return successFlag
        elif str(rts.split(',')[0])=='DELAY':
            print(str(sourceID)+': delaying sending model '+str(modelID))
            conn.sendall(('ACK,'+str(data.split(',')[1])).encode())
            # print(str(sourceID)+"S: ACK"+str(data.split(',')[1]))
            # print("delay in sending model: "+str(sourceID)+":"+str(data.split(',')[1]))
            return successFlag
        
    # print("exiting on checkSimModelsHandshake"+str(sourceID))    
    return 0

def receiveCSMData(sourceID,modelID,numPackets,lenofModelList,conn):
    modelList = receivePickledObject(sourceID,modelID,lenofModelList,conn,False)
    CSMdict = checkNeighboursCSM(sourceID,modelID,modelList)

    return sendCSMDetails(sourceID,modelID,CSMdict,conn)

def checkNeighboursCSM(sourceID,modelID,modelList):
    neigh = list(NETWORK.neighbors(sourceID))
    CSMdict=dict()
    # print(neigh)
    # print(modelList)
    # print("-------------------MODELS ALREADY SENT:"+str(REPLACEMENT))
    # print(MODELSSENT)
    if REPLACEMENT == 0:
        for n in neigh:
            CSMdict[n] =False
        # print("SEND TO ALLL: "+str(CSMdict))
        return CSMdict
    for n in neigh:
        CSMdict[n]=False
        for mID in modelList:
            # print(str(n)+" checking model ID: "+str(mID))
            if mID in MODELSSENT[n] or str(mID).split('-')[0]==n:
                CSMdict[n]=True
                break
    return CSMdict

def sendCSMDetails(sourceID,modelID,CSMdict,conn):
    print(str(sourceID)+': CSM model to '+str(modelID)+': '+str(CSMdict))
    successFlag = sendPickledObject(sourceID,modelID,CSMdict,conn,'CSMdict,')
    if successFlag:
        recACK = conn.recv(1024).decode()
        # print(str(sourceID)+"R: "+str(recACK))
        # print("recACK for CSMDetails "+str(sourceID)+":"+str(recACK))
        
        if str(recACK.split(',')[0])=='ACK':
            conn.sendall(('CSMDetailsComplete,'+str(modelID)).encode())
            # print(str(sourceID)+"S: CSMDetailsComplete"+str(modelID))
            return 1
    # print("exiting on sendCSMDetails"+str(sourceID))    
    return 0

def updateModelReplacementInfo(sourceID,sendReplaceSources,justReplaceSources,originalModelID,subModelID,subModel):
    global MODELSTOREPLACE
    global MODELSTORECEIVE
    # if '-' in originalModelID:
        # originalSourceID = originalModelID.split('-')[0]
    # if not(str(sourceID) in originalModelID):
    if not('-' in originalModelID):
        originalModelID = str(sourceID)+'-'+str(originalModelID)
    
    if sourceID not in originalModelID:
        print(str(sourceID)+" is trying to discard model "+str(originalModelID)+"!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        neigh = list(NETWORK.neighbors(sourceID))
        for n in neigh:
            if originalModelID in MODELSTORECEIVE[n].keys():
                # MODELSTOREPLACE[n][originalModelID] = 
                MODELSTOREPLACE[n][originalModelID]={'subModelID':subModelID,'replaced':False,'sendModel':False,'modelInfo':None}
                MODELSTORECEIVE[n][originalModelID]['replace']=True
            if subModelID not in MODELSTORECEIVE[n].keys() and (subModelID not in MODELSTOREPLACE[n].keys()) and n not in subModelID:
                MODELSTORECEIVE[n][subModelID]={'source':sourceID, 'originSource':str(subModelID.split('-')[0]), 'sent':False,'replace':False,'model':subModel}
                MODELSTOREPLACE[n][originalModelID]={'subModelID':subModelID,'replaced':False,'sendModel':True,'modelInfo':subModel}

def resolveSubModID(originalModelID,subModelID,source_ID):
    # global MODELSTOREPLACE
    # global MODELSTORECEIVE
    subModelSource = subModelID.split('-')[0]
    oldModelIDChain = [originalModelID]
    while subModelID in MODELSTOREPLACE[subModelSource].keys() and subModelID not in oldModelIDChain:
        oldModelIDChain.append(subModelID)
        currentSubIDSource = subModelSource
        subModelID = MODELSTOREPLACE[currentSubIDSource][subModelID]['subModelID']
    # if subModelID in originalModelID:
    return subModelID


def modelReplacementHandshake(sourceID,data,conn):
    global DISCARDEDMODELS
    REPLACEflag = data.split(',')[0]
    # print(str(sourceID)+"R: "+str(data))
    if REPLACEflag == 'REPLACE':
        originalModelID = data.split(',')[1]
        subModelID = data.split(',')[2]
        DISCARDEDMODELS[sourceID].append(originalModelID)
        # subModelID = resolveSubModID(originalModelID,subModelID,sourceID)
        sendReplace,justReplace = getReceiverReplacementLists(sourceID,originalModelID,subModelID)
        successFlag = sendReplaceDetails(sourceID,subModelID,sendReplace,conn)
        subModel=None
        print(str(sourceID)+': wants to replace model '+str(originalModelID)+' with '+str(subModelID))
        if successFlag:
            recACK = conn.recv(1024).decode()
            # print(str(sourceID)+"R: "+str(recACK))
            # print("recACK for ReplaceDetails "+str(sourceID)+":"+str(recACK))
            # print(str(recACK.split(',')[1]))
        
            if str(recACK.split(',')[1])=='False':
                # print("ReplaceListcomplete")
                print(str(sourceID)+': doesnt need to send replacement model '+str(subModelID))
                conn.sendall(('ReplaceListComplete,'+str(originalModelID)).encode())
                # print(str(sourceID)+"S: ReplaceListComplete "+str(originalModelID))
                # return 1
            elif str(recACK.split(',')[1])=='True':
                # print("true for replacement")
                print(str(sourceID)+': needs to send replacement model '+str(subModelID))
                subModel = receiveReplaceModel(sourceID,recACK,conn)
            updateModelReplacementInfo(sourceID,sendReplace,justReplace,originalModelID,subModelID,subModel)
                # return 1
        # print("successFlag at end of handshake is:" +str(successFlag))
    return successFlag




def getReceiverReplacementLists(sourceID,modelID,subModelID):
    neigh = list(NETWORK.neighbors(sourceID))
    sendReplaceList = []
    justReplaceList = []
    for n in neigh:
        if subModelID in MODELSTORECEIVE[n].keys() or n in subModelID:
            justReplaceList.append(n)
        else:
            sendReplaceList.append(n)
    # print("send and just replace lists")
    # print(justReplaceList)
    # print(sendReplaceList)
    return sendReplaceList,justReplaceList

def sendReplaceDetails(sourceID,subID,sendReplace,conn):
    return sendPickledObject(sourceID,subID,sendReplace,conn,'ReplaceList,')

def receiveReplaceModel(sourceID,ACKmsg,conn):
    submodelID = ACKmsg.split(',')[2]
    # print(str(sourceID)+"R: "+str(ACKmsg))
    numPackets = int(ACKmsg.split(',')[3])
    lenofModel = int(ACKmsg.split(',')[4])
    conn.sendall(('ACK,'+str(numPackets)).encode())
    # print(str(sourceID)+"S: ACK "+str(numPackets))
    subModel = receivePickledObject(sourceID,submodelID,lenofModel,conn,True)
    # print("received substitue: "+str(submodelID))
    return subModel
   


def receiveHandshake(sourceID,data,conn):
    RTSFlag = data.split(',')[0]
    # print(str(sourceID)+"R: "+str(RTSFlag))
    if 'RTS' in RTSFlag:
        modelID = data.split(',')[1]
        print(str(sourceID)+': is sending model '+str(modelID))
        # print(modelID)
        numPackets = int(data.split(',')[2])
        lenofModel = int(data.split(',')[3])
        lenReceiversPackets = int(data.split(',')[4])
        numReceiversPackets = int(data.split(',')[5])
        # print("HEREEEEEEEEEEEE:"+str(sourceID))
        conn.sendall(('ACK,'+str(numPackets)).encode())
        print(str(sourceID)+"S: ACK "+str(numPackets))
        receiverList = receivePickledObject(sourceID,modelID,lenReceiversPackets,conn)
        # print("receiver list"+str(sourceID))
        # print(receiverList)
        model = receivePickledObject(sourceID,modelID,lenofModel,conn)
        storeModel(sourceID,modelID,model,receiverList)
        # successfulRec = receiveData(sourceID,modelID,numPackets,lenofModel,conn)
        return 1
        
    # print("exiting on receiveHandshake"+str(sourceID))    
    return 0


def storeModel(sourceID,modelID,model,receiverList):
    global MODELS
    global MODELSTORECEIVE
    global MODELSTOREPLACE
    # model = pickle.loads(pickledModel)
    # print(sourceID, modelID)
    print(str(sourceID)+" storing: "+str(modelID))
    originSourceID = sourceID
    originModelID = modelID
    if '-' in str(modelID):
        originSourceID = str(modelID).split('-')[0]
        # originModelID = str(modelID).split('-')[1]
    else:
        modelID = str(sourceID)+'-'+str(modelID)
        if modelID not in MODELS[sourceID].keys():
            MODELS[sourceID][modelID] = model
    for r in receiverList:
        # print(r)
        if modelID not in MODELSTORECEIVE[r].keys():
            MODELSTORECEIVE[r][modelID]={'source':sourceID, 'originSource':originSourceID, 'sent':False,'replace':False,'model':model}
        if modelID in MODELSTOREPLACE[r].keys():
            if MODELSTOREPLACE[r][modelID]['replaced']==True:
                MODELSTORECEIVE[r][modelID]={'source':sourceID, 'originSource':originSourceID, 'sent':False,'replace':False,'model':model}
            else:
                MODELSTORECEIVE[r][modelID]['replace']=False
            del MODELSTOREPLACE[r][modelID]
            
    # print("storing model")
    # print(MODELSTORECEIVE)

    # print(sourceID, MODELS[sourceID])

def initiate(threadID,name,uid,PORT,learnerType,fp,inFile,outFile,sFrom,sTo,weightType,recievedModels,
        runID,nums,cullThresh,miThresh,paThresh,PCAVar,default_pred,frameworkType,connDeg,neighbors,replace):
    out = open(os.devnull,'w')
    # if ('PAC' in weightType) or weightType == 'OLSPAC' or weightType == 'OLSCL2' or weightType == 'OLSCL' or 'AWE' in weightType or 'AddExp' in weightType:
        # out = open(outFile,'w')
    # else:
        # out = open(os.devnull,'w')
    # out = open(os.devnull,'w')
    # if ('D004J007' in uid):# or ('D003J005' in uid) or ('SuddenT2' in uid):
        # out = open(outFile,'w')
    # out = open(outFile,'w')

    neighborStr = ""
    for i in neighbors:
        neighborStr += str(i)+","
    neighborStr = neighborStr[:-1]
    modelsSent = []
    # portIDs = [] 
    # socketHolders = []
    # numPorts = len(journeyList)
    # PORT = socketOffset
    connected = False
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    while (not connected) and PORT <50000:
        try:
            print("trying to bind to port: "+str(PORT))
            s.bind(('localhost',PORT))
            s.listen(1)
            connected = True
        except socket.error:
            print("incrementing port")
            PORT+=1
            connected=False
    if not connected:
        print("Failed to create socket2")
        s.close()
        s = None
    if s is None:
        print("exiting")
        sys.exit(1)

    if CD_TYPE == 'RePro':
        args = ["python3",fp,
                "--id", str(threadID),
                "--port",str(PORT),
                "--learner",str(learnerType),
                "--from",str(sFrom),
                "--to",str(sTo),
                "--fp",str(inFile),
                "--window",str(MAX_WINDOW),
                "--ReProAcc",str(MODEL_HIST_THRESHOLD_ACC),
                "--ReProProb",str(MODEL_HIST_THRESHOLD_PROB),
                "--ensemble",str(weightType),
                "--runid",str(runID),
                "--numStreams",str(nums),
                "--uid",str(uid),
                "--perfCull",str(cullThresh),
                "--miCull",str(miThresh),
                "--paCull",str(paThresh),
                "--variance",str(PCAVar),
                "--domain",str(default_pred),
                "--frameworkType",str(frameworkType),
                "--connDeg",str(connDeg),
                "--neighbors",str(neighborStr),
                "--replace",str(replace)]
    else:
        args = ["python3",fp,
                "--id", str(threadID),
                "--port",str(PORT),
                "--learner",str(learnerType),
                "--from",str(sFrom),
                "--to",str(sTo),
                "--fp",str(inFile),
                "--window",str(MAX_WINDOW),
                "--ReProAcc",str(MODEL_HIST_THRESHOLD_ACC),
                "--ReProProb",str(MODEL_HIST_THRESHOLD_PROB),
                "--ADWINDelta",str(ADWIN_DELTA),
                "--ensemble",str(weightType),
                "--runid",str(runID),
                "--numStreams",str(nums),
                "--uid",str(uid),
                "--perfCull",str(cullThresh),
                "--miCull",str(miThresh),
                "--paCull",str(paThresh),
                "--variance",str(PCAVar),
                "--domain",str(default_pred),
                "--frameworkType",str(frameworkType),
                "--connDeg",str(connDeg),
                "--neighbors",str(neighborStr),
                "--replace",str(replace)]
    p = subprocess.Popen(args,stdout=out)
    conn,addr = s.accept()
    source_ID = conn.recv(1024).decode()
    # print("connected to: "+repr(source_ID))
    conn.sendall(("connected ACK").encode())
    
    while 1:
        # listen for rts
        # do handshake
        # receive model
        data = conn.recv(1024).decode()
        print(repr(data))
        flag = data.split(',')[0]
        if flag == 'CSM':
            # print("CSM________________________________________ for:"+str(source_ID))
            successFlag = checkSimModelsHandshake(uid,data,conn)
            # print("end CSM for:"+str(source_ID))
        elif flag == 'RTS':
            # # successFlag = receiveHandshake(threadID,data,conn)
            # print("RTS for:"+str(source_ID))
            successFlag = receiveHandshake(uid,data,conn)
            # print("endRTS for:"+str(source_ID))
        elif flag == 'RTR':
            # successFlag, modelsSent = sendHandshake(threadID,data,conn,modelsSent)
            # print("RTR for:"+str(source_ID))
            successFlag, modelsSent = sendHandshake(uid,data,conn,modelsSent)
            # print("endRTR for:"+str(source_ID))
        elif flag == 'REPLACE':
            # print("REPLACE________________________________________ for:"+str(source_ID))
            successFlag = modelReplacementHandshake(uid,data,conn)
        else:
            # print("flag recieved is not RTR or RTS")
            successFlag = 0

        #send ACK
        # print(str(source_ID)+"connection established with: "+repr(data))
        if not data: break
        if not successFlag:
            print("communication FAIL!!!!!!!!!!!!!!!!!!")
            print(str(uid)+"failed on: "+str(data))
            exit()
            break
        time.sleep(1)
        
    p.wait()
    conn.close()
    s.close()
    out.close()

def getHeatingDates():
    dates = dict()
    dates = {
            0:{'start':"2014-01-01",'end':"2015-03-31"},
            1:{'start':"2015-01-01",'end':"2015-12-31"},
            2:{'start':"2014-09-01",'end':"2015-03-30"},
            3:{'start':"2015-01-01",'end':"2015-09-30"},
            4:{'start':"2014-01-01",'end':"2015-06-30"}
            }
    testdates = {
            0:{'start':"2014-03-01",'end':"2014-08-31"},
            1:{'start':"2015-01-01",'end':"2015-06-30"}}
    # return testdates
    return dates

def getFPdict(driftType):
    FPdict = {
            # (str(driftType)+'S1'):'../../HyperplaneDG/Data/Datastreams/SOURCEMultiConcept'+str(driftType)+'.csv',
            (str(driftType)+'T1'):'../HyperplaneDG/Data/Datastreams/TARGET1MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T2'):'../HyperplaneDG/Data/Datastreams/TARGET2MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T3'):'../HyperplaneDG/Data/Datastreams/TARGET3MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T4'):'../HyperplaneDG/Data/Datastreams/TARGET4MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T5'):'../HyperplaneDG/Data/Datastreams/TARGET5MultiConcept'+str(driftType)+'1.csv'}
    testFPdict = {
            # (str(driftType)+'S1'):'../../HyperplaneDG/Data/Datastreams/SOURCEMultiConcept'+str(driftType)+'.csv',
            (str(driftType)+'T1'):'../HyperplaneDG/Data/Datastreams/TARGET1MultiConcept'+str(driftType)+'1.csv',
            (str(driftType)+'T2'):'../HyperplaneDG/Data/Datastreams/TARGET2MultiConcept'+str(driftType)+'1.csv'}
    # return testFPdict
    return FPdict

def get7FollowingFPdict():
    testFPdict = {
            'D001J001': '../FollowingDistanceData/dr001J001.csv',
            'D001J003': '../FollowingDistanceData/dr001J003.csv',#}
            # 'D001J002': '../FollowingDistanceData/dr001J002.csv',
            # 'D001J003': '../FollowingDistanceData/dr001J003.csv',
            # 'D002J001': '../FollowingDistanceData/dr002J001.csv',
            # 'D002J002': '../FollowingDistanceData/dr002J002.csv',
            # 'D002J003': '../FollowingDistanceData/dr002J003.csv',
            # 'D003J002': '../FollowingDistanceData/dr003J002.csv',
            'D003J005': '../FollowingDistanceData/dr003J005.csv',
            'D003J006': '../FollowingDistanceData/dr003J006.csv',
            'D004J003': '../FollowingDistanceData/dr004J003.csv',
            'D004J005': '../FollowingDistanceData/dr004J005.csv',
            'D004J006': '../FollowingDistanceData/dr004J006.csv'}#,
            # 'D004J007': '../FollowingDistanceData/dr004J007.csv',
            # 'D004J020': '../FollowingDistanceData/dr004J020.csv',
    return testFPdict


def getFollowingFPdict():
    FPdict = {
            'D001J001': '../FollowingDistanceData/dr001J001.csv',
            'D001J002': '../FollowingDistanceData/dr001J002.csv',
            'D001J003': '../FollowingDistanceData/dr001J003.csv',
            'D002J001': '../FollowingDistanceData/dr002J001.csv',
            'D002J002': '../FollowingDistanceData/dr002J002.csv',
            'D002J003': '../FollowingDistanceData/dr002J003.csv'}
    testFPdict = {
            'D001J001': '../FollowingDistanceData/dr001J001.csv',
            'D001J003': '../FollowingDistanceData/dr001J003.csv'}

    # return testFPdict
    return FPdict

def main():
    global MODELS
    global MODELSTORECEIVE
    global MODELSSENT
    global MODELSTOREPLACE
    global DISCARDEDMODELS
    global INIT_DAYS#80
    global MODEL_HIST_THRESHOLD_PROB# 0.4
    global MAX_WINDOW#80
    global STABLE_SIZE#2* MAX_WINDOW
    global MODEL_HIST_THRESHOLD_ACC#0.5
    global THRESHOLD#0.5
    global DEFAULT_PRED
    global CD_TYPE
    global ADWIN_DELTA
    global NETWORK
    global NODE_ID_KEYS
    global REPLACEMENT
    time.sleep(random.random()*5)
    parser = OptionParser(usage="usage: prog options",version="BOTL v2.0")
    parser.add_option("-d","--domain",default = "Following",dest="DEFAULT_PRED",help="domain: Following, Heating, Sudden, Gradual")
    parser.add_option("-t","--type",default = "RePro",dest= "CD_TYPE",help="Concept Drift Type: RePro, ADWIN, AWPro")
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
    parser.add_option("-l","--learner",default="SVR",dest="learnerType",help="Type of model used by the underlying CDD")
    parser.add_option("-g","--connectivity",default=0,dest="degConnectivity",help="Degree of network connectivity (fully connected = 0)")
    parser.add_option("-b","--replace",default=0,dest="replace",help="1 is with model replacement, 0 is without")
    sourceInfo = dict()
    targetInfo = dict()
    (options,args) = parser.parse_args()
    
    DEFAULT_PRED=str(options.DEFAULT_PRED)
    CD_TYPE=str(options.CD_TYPE)
    MAX_WINDOW=int(options.MAX_WINDOW)
    MODEL_HIST_THRESHOLD_ACC=float(options.MODEL_HIST_THRESHOLD_ACC)
    MODEL_HIST_THRESHOLD_PROB=float(options.MODEL_HIST_THRESHOLD_PROB)
    runID=int(options.runID)
    numStreams=int(options.numStreams)
    ADWIN_DELTA=float(options.ADWIN_DELTA)
    socketOffset=int(options.socketOffset)
    weightType=str(options.weightType)
    CThresh=float(options.CThresh)
    MThresh=float(options.MThresh)
    PAThresh=float(options.PAThresh)
    PCAVar=float(options.PCAVar)
    REPLACEMENT = int(options.replace)
    connectivity = int(options.degConnectivity)
    frameworkType = ""
    if "," in str(options.learnerType):
        LEARNER_TYPE = [str(i) for i in str(options.learnerType).split(',')]
        frameworkType += "MixedLearners"
        for l in LEARNER_TYPE:
            frameworkType += ("_"+str(l))
    else:
        LEARNER_TYPE = [str(options.learnerType)]
        frameworkType+="Learner"
        frameworkType += ("_"+str(LEARNER_TYPE[0]))
    INIT_DAYS = MAX_WINDOW#80
    STABLE_SIZE = 2* MAX_WINDOW
    # MODEL_HIST_THRESHOLD_PROB = 0.4
    THRESHOLD = MODEL_HIST_THRESHOLD_ACC# = THRESHOLD#0.5
    REPLACEMENTSTR = ""
    if REPLACEMENT:
        REPLACEMENTSTR='replace'

    FPdict = dict()
    
    if DEFAULT_PRED == 'Following':
        if numStreams == 0:
            FPdict = get7FollowingFPdict()
            # numStreams = 7
        else:
            FPdict = getFollowingFPdict()
        # FPdict = getFollowingFPdict()
    elif DEFAULT_PRED == 'Heating':
        FPdict = getHeatingDates()
    else:
        FPdict = getFPdict(DEFAULT_PRED)
    journeyList = list(FPdict.keys())
    random.shuffle(journeyList)
    
    if numStreams == 0:
        journeyList = journeyList[0:7]
    else:
        journeyList = journeyList[0:numStreams]

    # nodeList = [str(LEARNER_TYPE)+"_"+str(j) for j in journeyList]
    print(LEARNER_TYPE)
    LEARNER_TYPE = np.resize(LEARNER_TYPE,len(journeyList)).tolist()
    random.shuffle(LEARNER_TYPE)
    # LEARNER_TYPE = ['SVR','Ridge','SVR','Ridge','Ridge','Ridge']
    nodeList = [str(LEARNER_TYPE[jdx%len(LEARNER_TYPE)])+"_"+str(j) for jdx,j in enumerate(journeyList)]
    
    if connectivity ==0:
        NETWORK = nx.complete_graph(len(nodeList))
        frameworkType += "_FullConn"
    else:
        NETWORK = nx.expected_degree_graph([connectivity for i in nodeList],selfloops=False)
        while(not nx.is_connected(NETWORK)):
            print("rerunning")
            NETWORK = nx.expected_degree_graph([connectivity for i in nodeList],selfloops=False)
        frameworkType += "_PartConnDeg"+str(connectivity)
    NODE_ID_KEYS = dict(enumerate(nodeList))
    NETWORK = nx.relabel_nodes(NETWORK,NODE_ID_KEYS,copy=False)
    for ni in NETWORK.nodes():
        print(ni)
        print(list(NETWORK.neighbors(ni)))


    for idx, i in enumerate(journeyList):
        source = dict()
        source['Name'] = "source"+str(idx)+":"+str(i)
        # thisLearnerType = str(LEARNER_TYPE)+'_'+str(i)
        thisLearnerType = str(LEARNER_TYPE[idx%len(LEARNER_TYPE)])
        source['uid'] = str(thisLearnerType)+'_'+str(i)
        # source['stdo']="TestResultsLog/Run"+str(runID)+"/"+str(LEARNER_TYPE)+str(weightType)+str(source['Name'])+str(numStreams)+"Out.txt"
        source['stdo']="TestResultsLog/Run"+str(runID)+"/"+str(REPLACEMENTSTR)+str(thisLearnerType)+str(weightType)+str(source['Name'])+str(numStreams)+"Out.txt"
        source['PORT'] = socketOffset+idx#portIDs[idx]#socketOffset+idx
        # source['Run'] = "source"+str(CD_TYPE)+".py"
        source['Run'] = "source"+str(CD_TYPE)+"2.py"
        # source['stdin'] = FPdict[i]
        source['weightType'] = weightType
        source['cullThresh'] = CThresh
        source['miThresh'] = MThresh
        source['paThresh']= PAThresh
        source['default_pred'] = DEFAULT_PRED
        source['PCAVar'] = PCAVar
        # source['learnerType']=LEARNER_TYPE
        source['learnerType']=thisLearnerType
        source['frameworkType']=frameworkType
        source['connDeg']=NETWORK.degree(source['uid'])
        source['neighbors']=list(NETWORK.neighbors(source['uid']))
        source['replace']=REPLACEMENT
        if DEFAULT_PRED == 'Heating':
            source['sFrom'] = FPdict[idx]['start']#startDates[i]#"2014-01-01"
            source['sTo'] = FPdict[idx]['end']#endDates[i]#"2014-02-28"
            source['stdin'] = "../HeatingSimDG/HeatingSimData/userSOURCEDataSimulation.csv"
        else:
            # source['sFrom'] = 470
            source['sFrom'] = 0
            source['sTo'] = 15000
            # source['sTo'] = 5000
            # source['sTo'] = 1000
            source['stdin'] = FPdict[i]

        sourceModels = dict()
        toRecModels = dict()
        toReplaceModels = dict()
        MODELS[nodeList[idx]] = sourceModels
        MODELSTORECEIVE[nodeList[idx]] = toRecModels
        MODELSSENT[nodeList[idx]] = []
        MODELSTOREPLACE[nodeList[idx]]=toReplaceModels
        DISCARDEDMODELS[nodeList[idx]]=[]
        receivedModels = []
        sourceInfo[idx] = source

    print("creating threads")
    totalTime = 0
    for k,v in sourceInfo.items():
        print(k, v)
        print("making thread")
        sThread = myThread(k,v,receivedModels,runID,numStreams)
        print("starting thread")
        sThread.start()
        totalTime = 0
        while not MODELS[nodeList[k]]:
            tts = random.uniform(0.0,1.2)
            # time.sleep(10)
            time.sleep(5)
            totalTime = totalTime+ 100
            if totalTime >= 300:
                print(" no stable models in :" +str(k))
                break
        print(" RECIEVED FIRST MODEL SO STARTING NEXT THREAD")
    # print(MODELSTORECEIVE)

if __name__ == '__main__':main()

