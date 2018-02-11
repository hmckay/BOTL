import numpy as np
import subprocess
import socket

import source
import pandas as pd
import pickle
import threading
import time

MODELS = dict()
INIT_DAYS = 30
STABLE_SIZE = 60
MAX_WINDOW = 30
MODEL_HIST_THRESHOLD_PROB = 0.1
MODEL_HIST_THRESHOLD_ACC = 0.55
THRESHOLD = 0.52


FILEstdin = {2000: '../../../HyperplaneDG/ICMLData/SOURCEMultiConceptSudden.csv', 
        2001: '../../../HyperplaneDG/ICMLData/TARGET1MultiConceptSudden5.csv',
        2002: '../../../HyperplaneDG/ICMLData/TARGET2MultiConceptSudden5.csv',
        2003: '../../../HyperplaneDG/ICMLData/TARGET3MultiConceptSudden5.csv',
        2004: '../../../HyperplaneDG/ICMLData/TARGET4MultiConceptSudden5.csv',
        2005: '../../../HyperplaneDG/ICMLData/TARGET5MultiConceptSudden5.csv'}


class myThread (threading.Thread):
    def __init__(self,threadID,info,receivedModels):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = info['Name']
        self.outputFile = info['stdo']
        self.PORT = info['PORT']
        self.fp = info['Run']
        self.inputFile = info['stdin']
        self.sFrom = info['sFrom']
        self.sTo = info['sTo']
        self.weightType = info['weightType']
        self.receivedModels = receivedModels
    
    def run(self):
        print "starting "+ self.name
        initiate(self.threadID,self.name,self.PORT,self.fp,self.inputFile,self.outputFile,self.sFrom,self.sTo,self.weightType,self.receivedModels)
        print "exiting " + self.name

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
        print "finished sending"
    recACK = conn.recv(1024)
    if modelID == recACK.split(',')[1]:
        modelsSent.append(modelID)
        print "models sent: "+str(modelsSent)
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

def initiate(threadID,name,PORT,fp,inFile,outFile,sFrom,sTo,weightType,recievedModels):
    #print outFile
    out = open(outFile,"a")
    modelsSent = []
    args = ['python',fp,str(threadID),str(PORT),str(sFrom),str(sTo),inFile,str(INIT_DAYS),str(MODEL_HIST_THRESHOLD_ACC), 
            str(MODEL_HIST_THRESHOLD_PROB),str(STABLE_SIZE),str(MAX_WINDOW),str(THRESHOLD),str(weightType)]
    p = subprocess.Popen(args,stdout=out)
    try: 
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.bind(('localhost',PORT))
        s.listen(1)
    except socket.error:
        print "Failed to create socket"
        s.close()
        s = None
    if s is None:
        print "exiting"
        sys.exit(1)
    conn,addr = s.accept()
    source_ID = conn.recv(1024)
    print "connected to: "+repr(source_ID)
    conn.sendall("connected ACK")
    
    while 1:
        # listen for rts
        # do handshake
        # receive model
        data = conn.recv(1024)
        print repr(data)
        flag = data.split(',')[0]
        if flag == 'RTS':
            successFlag = receiveHandshake(threadID,data,conn)
        elif flag == 'RTR':
            successFlag, modelsSent = sendHandshake(threadID,data,conn,modelsSent)
        else:
            print "flag recieved is not RTR or RTS"
            successFlag = 0

        #send ACK
        print "connection established with: "+repr(data)
        if not data: break
        if not successFlag:
            print "communication FAIL"
            break
        time.sleep(1)
        
    p.wait()
    conn.close()
    s.close()

def main():
    global MODELS
    sourceInfo = dict()
    targetInfo = dict()

    for i in range(2000,2006):
        source = dict()
        source['Name'] = "source"+str(i)
        source['stdo']="Results/5multipleSudden"+str(i)+"Out.txt"
        source['PORT'] = i
        source['Run'] = "source.py"
        source['stdin'] = FILEstdin[i]
        source['sFrom'] = 1
        source['sTo'] = 10000
        source['weightType'] = 'OLS'

        sourceModels = dict()
        MODELS[i] = sourceModels
        receivedModels = []
        sourceInfo[i] = source

    print "creating threads"

    for k,v in sourceInfo.iteritems():
        print k, v
        print "making thread"
        sThread = myThread(k,v,receivedModels)
        print "starting thread"
        sThread.start()
        while not MODELS[k]:
            print "waiting"
            time.sleep(5)
    

if __name__ == '__main__':main()

