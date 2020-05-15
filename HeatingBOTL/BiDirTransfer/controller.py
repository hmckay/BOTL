############################################################################
# Code for BOTL controller. This manages the source domains, and transfers 
# models between them. This implemetation uses RePro as the underlying drift 
# detector.
############################################################################
import numpy as np
import subprocess
import socket
#from sklearn.linear_model import SGDRegressor as SGD
#from sklearn.kernel_approximation import RBFSampler as RBF
#from sklearn.pipeline import Pipeline
import source
import pandas as pd
import pickle
import threading
import time
from datetime import datetime
import sys
import random
import os

MODELS = dict()
INIT_DAYS = 0#80
MODEL_HIST_THRESHOLD_PROB = 0# 0.4
MAX_WINDOW = 0#80
STABLE_SIZE = 0#2* MAX_WINDOW
MODEL_HIST_THRESHOLD_ACC = 0#0.5
THRESHOLD = 0#0.5
# INIT_DAYS = 14
# STABLE_SIZE = 14
# MAX_WINDOW = 14
# MODEL_HIST_THRESHOLD_PROB = 0.66
# MODEL_HIST_THRESHOLD_ACC = 0.55#48
# THRESHOLD = 0.64


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
    # def __init__(self,threadID,info,receivedModels,sourceFlag):
        # threading.Thread.__init__(self)
        # self.threadID = threadID
        # self.name = info['Name']
        # self.outputFile = info['stdo']
        # self.PORT = info['PORT']
        # self.fp = info['Run']
        # self.inputFile = info['stdin']
        # self.sFrom = info['sFrom']
        # self.sTo = info['sTo']
        # self.weightType = info['weightType']
        # self.receivedModels = receivedModels
    
    def run(self):
        print("starting "+ self.name)
        # initiate(self.threadID,self.name,self.PORT,self.fp,self.inputFile,self.outputFile,self.sFrom,self.sTo,self.weightType,self.receivedModels)
        initiate(self.threadID,self.name,self.uid,self.PORT,self.fp,self.inputFile,self.outputFile,self.
                sFrom,self.sTo,self.weightType,self.receivedModels,self.runID,self.numStreams,
                self.cullThresh,self.miThresh)
        print("exiting " + self.name)

def getModelsToSend(threadID,modelsSent):
    toSend = dict()
    allModels = MODELS
    for tID,modelDict in allModels.items():
        if tID != threadID:
            for modelID,model in modelDict.items():
                sourceModID = str(tID)+'-'+str(modelID)
                print("sourceModID: "+str(sourceModID))
                print("modelsSent: "+str(modelsSent))
                if sourceModID not in modelsSent:
                    toSend[sourceModID] = model
    return toSend


def sendHandshake(targetID,data,conn,modelsSent):
    RTRFlag = data.split(',')[0]
    if RTRFlag == 'RTR':
        target_ID = int(data.split(',')[1])
        if target_ID != targetID:
            print("changed targetIDs")
            return 0,modelsSent
        #get number of models to send
        modelsToSend = getModelsToSend(targetID,modelsSent)
        numModels = len(modelsToSend)
        conn.sendall(('ACK,'+str(numModels)).encode())
        ack = conn.recv(1024).decode()
        print(targetID, repr(ack))
        if ack == 'ACK':
            return sendModels(targetID,numModels,modelsToSend,modelsSent,conn)
        elif ack == 'END':
            return 1,modelsSent
    return 0,modelsSent

def sendModels(targetID,numModels,modelsToSend,modelsSent,conn):
    for modelID,model in modelsToSend.items():
        modelToSend = pickle.dumps(model)
        print("modelID is: "+str(modelID))
        #print "pickled model len at controller is: " + str(len(modelToSend))
        brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
        numPackets = len(brokenBytes)
        lenofModel = len(modelToSend)
        print(str(targetID)+"BROKEN BYTES LEN: "+str(numPackets))
        conn.sendall(('RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel)).encode())
        ack = conn.recv(1024).decode()
        ackNumPackets = int(ack.split(',')[1])
        print("acked number of packets is " +str(ackNumPackets))

        if ackNumPackets == numPackets:
            flag,modelsSent = sendModel(modelID,brokenBytes,modelsSent,conn)
        else:
            print("failed to send model: "+str(modelID))
            return 0,modelsSent
    return 1,modelsSent

def sendModel(modelID,brokenBytes,modelsSent,conn):
    #f = open("temp"+str(modelID)+".txt",'a')
    for idx,i in enumerate(brokenBytes):
        #f.write("iter: "+str(idx)+"\n")
        #f.write(str(i))
        #f.write("\n\n")
        conn.sendall(i)
        #print "finished sending"
    recACK = conn.recv(1024).decode()
    if modelID == recACK.split(',')[1]:
        modelsSent.append(modelID)
        print("models sent: "+str(modelsSent))
        #modelsSent = updateSentModels(modelID,modelsSent)
        return 1, modelsSent
    return 0, modelsSent


def receiveHandshake(sourceID,data,conn):
    RTSFlag = data.split(',')[0]
    if RTSFlag == 'RTS':
        modelID = data.split(',')[1]
        print(modelID)
        numPackets = int(data.split(',')[2])
        lenofModel = int(data.split(',')[3])

        conn.sendall(('ACK,'+str(numPackets)).encode())

        return receiveData(sourceID,modelID,numPackets,lenofModel,conn)
        
    return 0

def receiveData(sourceID,modelID,numPackets,lenofModel,conn):
    # send ACK
    pickledModel = b''
    while (len(pickledModel)<lenofModel):
        #for i in range(0,numPackets):
        pickledModel = pickledModel + conn.recv(1024)
    conn.sendall(('RECEIVED,'+str(modelID)).encode())

    storeModel(sourceID,modelID,pickledModel)
    return 1

def storeModel(sourceID,modelID,pickledModel):
    global MODELS
    model = pickle.loads(pickledModel)
    print(sourceID, modelID)
    MODELS[sourceID][modelID] = model

    print(sourceID, MODELS[sourceID])

'''
def getModelsToSend(threadID,modelsSent):
    toSend = dict()
    allModels = MODELS
    for tID,modelDict in allModels.iteritems():
        if tID != threadID:
            for modelID,model in modelDict.iteritems():
                sourceModID = str(tID)+'-'+str(modelID)
                print "sourceModID: "+str(sourceModID)
                print "modelsSent: "+str(modelsSent)
                if sourceModID not in modelsSent:
                    toSend[sourceModID] = model
    return toSend


def sendHandshake(targetID,data,conn,modelsSent):
    RTRFlag = data.split(',')[0]
    if RTRFlag == 'RTR':
        target_ID = int(data.split(',')[1])
        if target_ID != targetID:
            print "changed targetIDs"
            return 0,modelsSent
        #get number of models to send
        modelsToSend = getModelsToSend(targetID,modelsSent)
        numModels = len(modelsToSend)
        conn.sendall('ACK,'+str(numModels))
        ack = conn.recv(1024)
        print targetID, repr(ack)
        if ack == 'ACK':
            return sendModels(targetID,numModels,modelsToSend,modelsSent,conn)
        elif ack == 'END':
            return 1,modelsSent
    return 0,modelsSent

def sendModels(targetID,numModels,modelsToSend,modelsSent,conn):
    for modelID,model in modelsToSend.iteritems():
        modelToSend = pickle.dumps(model)
        brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
        numPackets = len(brokenBytes)
        conn.sendall('RTS,'+str(modelID)+','+str(numPackets))
        ack = conn.recv(1024)
        ackNumPackets = int(ack.split(',')[1])

        if ackNumPackets == numPackets:
            flag,modelsSent = sendModel(modelID,brokenBytes,modelsSent,conn)
        else:
            print "failed to send model: "+str(modelID)
            return 0,modelsSent
    return 1,modelsSent

def sendModel(modelID,brokenBytes,modelsSent,conn):
    for i in brokenBytes:
        conn.sendall(i)
        #print "finished sending"
    recACK = conn.recv(1024)
    if modelID == recACK.split(',')[1]:
        modelsSent.append(modelID)
        print "models sent: "+str(modelsSent)
        #modelsSent = updateSentModels(modelID,modelsSent)
        return 1, modelsSent
    return 0, modelsSent


def receiveHandshake(sourceID,data,conn):
    RTSFlag = data.split(',')[0]
    if RTSFlag == 'RTS':
        modelID = data.split(',')[1]
        print modelID
        numPackets = int(data.split(',')[2])

        conn.sendall('ACK,'+str(numPackets))

        return receiveData(sourceID,modelID,numPackets,conn)
        
    return 0

def receiveData(sourceID,modelID,numPackets,conn):
    # send ACK
    pickledModel = ""
    for i in range(0,numPackets):
        pickledModel = pickledModel + conn.recv(1024)
    conn.sendall('RECEIVED,'+str(modelID))

    storeModel(sourceID,modelID,pickledModel)
    return 1

def storeModel(sourceID,modelID,pickledModel):
    global MODELS
    model = pickle.loads(pickledModel)
    print sourceID, modelID
    MODELS[sourceID][modelID] = model

    print sourceID, MODELS[sourceID]
'''
# def initiate(threadID,name,PORT,fp,inFile,outFile,sFrom,sTo,weightType,recievedModels):
def initiate(threadID,name,uid,PORT,fp,inFile,outFile,sFrom,sTo,weightType,recievedModels,runID,nums,cullThresh,miThresh):
    out = open(os.devnull,'w')
    # if weightType == 'OLSPAC2' or weightType == 'OLSPAC' or weightType == 'OLSCL2' or weightType == 'OLS':
        # out = open(outFile,'w')
    # else:
        # out = open(os.devnull,'w')

    modelsSent = []
    args = ['python3',fp,str(threadID),str(PORT),str(sFrom),str(sTo),inFile,str(INIT_DAYS),str(MODEL_HIST_THRESHOLD_ACC), 
            str(MODEL_HIST_THRESHOLD_PROB),str(STABLE_SIZE),str(MAX_WINDOW),str(THRESHOLD),str(weightType),str(runID),
            str(nums),str(uid),str(cullThresh),str(miThresh)]
    # out = open(outFile,"a")
    # modelsSent = []
    # args = ['python3',fp,str(threadID),str(PORT),str(sFrom),str(sTo),inFile,str(INIT_DAYS),str(MODEL_HIST_THRESHOLD_ACC), 
            # str(MODEL_HIST_THRESHOLD_PROB),str(STABLE_SIZE),str(MAX_WINDOW),str(THRESHOLD),str(weightType)]
    p = subprocess.Popen(args,stdout=out)
    try: 
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.bind(('localhost',PORT))
        s.listen(1)
    except socket.error:
        print("Failed to create socket")
        s.close()
        s = None
    if s is None:
        print("exiting")
        sys.exit(1)
    conn,addr = s.accept()
    source_ID = conn.recv(1024).decode()
    print("connected to: "+repr(source_ID))
    conn.sendall(("connected ACK").encode())
    
    while 1:
        # listen for rts
        # do handshake
        # receive model
        data = conn.recv(1024).decode()
        print(repr(data))
        flag = data.split(',')[0]
        if flag == 'RTS':
            successFlag = receiveHandshake(threadID,data,conn)
        elif flag == 'RTR':
            successFlag, modelsSent = sendHandshake(threadID,data,conn,modelsSent)
        else:
            print("flag recieved is not RTR or RTS")
            successFlag = 0

        #send ACK
        print("connection established with: "+repr(data))
        if not data: break
        if not successFlag:
            print("communication FAIL")
            break
        time.sleep(1)
        #else: recieveData(source_ID,s,conn,addr)
        
    p.wait()
    conn.close()
    s.close()
    out.close()

def getDates():
    dates = dict()
    dates = {
            0:{'start':"2014-01-01",'end':"2015-03-31"},
            1:{'start':"2015-01-01",'end':"2015-12-31"},
            2:{'start':"2014-09-01",'end':"2015-03-30"},
            3:{'start':"2015-01-01",'end':"2015-09-30"},
            4:{'start':"2014-01-01",'end':"2015-06-30"}
            }
    return dates
    # startDates = {2000: "2014-01-01", 2001:"2015-01-01", 2002:"2014-09-01",2003:"2015-01-01",2004:"2014-01-01"}
    # endDates = {2000: "2015-03-31",2001:"2015-12-31",2002:"2015-03-30",2003:"2015-09-30",2004:"2015-06-30"}

def main():
    global MODELS
    global INIT_DAYS#80
    global MODEL_HIST_THRESHOLD_PROB# 0.4
    global MAX_WINDOW#80
    global STABLE_SIZE#2* MAX_WINDOW
    global MODEL_HIST_THRESHOLD_ACC#0.5
    global THRESHOLD#0.5
    sourceInfo = dict()
    targetInfo = dict()
    runID = int(sys.argv[1])
    numStreams = int(sys.argv[2])
    socketOffset = int(sys.argv[3])
    weightType = str(sys.argv[4])
    CThresh = float(sys.argv[5])
    MThresh = float(sys.argv[6])
    MAX_WINDOW = int(sys.argv[7])#80
    INIT_DAYS = MAX_WINDOW#80
    STABLE_SIZE = 2* MAX_WINDOW
    MODEL_HIST_THRESHOLD_PROB = 0.4
    THRESHOLD = float(sys.argv[8])#0.5
    MODEL_HIST_THRESHOLD_ACC = THRESHOLD#0.5
    # weightType = str(sys.argv[1])
    # offset = int(sys.argv[2])
    # sourceInfo = dict()
    # targetInfo = dict()
    # startDates = {2000: "2014-01-01", 2001:"2015-01-01"}#, 2002:"2014-09-01",2003:"2015-01-01",2004:"2014-01-01"}
    # endDates = {2000: "2015-03-31",2001:"2015-12-31"}#,2002:"2015-03-30",2003:"2015-09-30",2004:"2015-06-30"}
    # startDates = {2000: "2014-01-01", 2001:"2015-01-01", 2002:"2014-09-01",2003:"2015-01-01",2004:"2014-01-01"}
    # endDates = {2000: "2015-03-31",2001:"2015-12-31",2002:"2015-03-30",2003:"2015-09-30",2004:"2015-06-30"}
    users = getDates()

    for idx,val in enumerate(users):#startDates.items():
        source = dict()
        source['Name'] = "source"+str(idx)
        source['uid'] = str(idx)
        source['stdo']="FEOLSFINALsourceFullSim"+str(idx)+"Out.txt"
        source['PORT'] = socketOffset+idx#i+(offset*1000)
        source['Run'] = "source.py"
        source['stdin'] = "../../HeatingSimDG/HeatingSimData/userSOURCEDataSimulation.csv"
        source['sFrom'] = users[idx]['start']#startDates[i]#"2014-01-01"
        source['sTo'] = users[idx]['end']#endDates[i]#"2014-02-28"
        source['weightType'] = weightType
        source['cullThresh'] = CThresh
        source['miThresh'] = MThresh

        sourceModels = dict()
        # MODELS[i+(offset*1000)] = sourceModels
        MODELS[idx] = sourceModels
        receivedModels = []
        # sourceInfo[i+(offset*1000)] = source
        sourceInfo[idx] = source

    # print("creating threads")

    print("creating threads")
    totalTime = 0
    for k,v in sourceInfo.items():
        print(k, v)
        print("making thread")
        sThread = myThread(k,v,receivedModels,runID,numStreams)
        print("starting thread")
        sThread.start()
        totalTime = 0
        while not MODELS[k]:
            tts = random.uniform(0.0,1.2)
            time.sleep(10)
            totalTime = totalTime+ 100
            if totalTime >= 300:
                print(" no stable models in :" +str(k))
                break
        print(" RECIEVED FIRST MODEL SO STARTING NEXT THREAD")
    # for k,v in sourceInfo.items():
        # print(k, v)
        # print("making thread")
        # sThread = myThread(k,v,receivedModels,1)
        # print("starting thread")
        # sThread.start()
        # totalTime=0
        # while not MODELS[k]:
            # time.sleep(10)
            # # totalTime = totalTime+ 100
            # # if totalTime >= 300:
                # # print " no stable models in :" +str(k)
                # # break
            # print("waiting")
            # #time.sleep(5)
        # #time.sleep(2) 
    

if __name__ == '__main__':main()

