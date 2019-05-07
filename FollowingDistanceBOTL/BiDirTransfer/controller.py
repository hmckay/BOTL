import numpy as np
import subprocess
import socket
import random
#from sklearn.linear_model import SGDRegressor as SGD
#from sklearn.kernel_approximation import RBFSampler as RBF
#from sklearn.pipeline import Pipeline
import source
import pandas as pd
import pickle
import threading
import time
import sys
import os

MODELS = dict()
INIT_DAYS = 80
MODEL_HIST_THRESHOLD_PROB = 0.4
MAX_WINDOW = 80
STABLE_SIZE = 2* MAX_WINDOW
MODEL_HIST_THRESHOLD_ACC = 0.5
THRESHOLD = 0.5

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
        self.sFrom = 0
        self.sTo = 10000
        self.weightType = info['weightType']
        self.cullThresh = info['cullThresh']
        self.miThresh = info['miThresh']
        self.receivedModels = receivedModels
        self.runID = runnum
        self.numStreams = nums
    
    def run(self):
        print "starting "+ self.name
        initiate(self.threadID,self.name,self.uid,self.PORT,self.fp,self.inputFile,self.outputFile,self.
                sFrom,self.sTo,self.weightType,self.receivedModels,self.runID,self.numStreams,
                self.cullThresh,self.miThresh)
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
        print "modelID is: "+str(modelID)
        #print "pickled model len at controller is: " + str(len(modelToSend))
        brokenBytes = [modelToSend[i:i+1024] for i in range(0,len(modelToSend),1024)]
        numPackets = len(brokenBytes)
        lenofModel = len(modelToSend)
        print str(targetID)+"BROKEN BYTES LEN: "+str(numPackets)
        conn.sendall('RTS,'+str(modelID)+','+str(numPackets)+','+str(lenofModel))
        ack = conn.recv(1024)
        ackNumPackets = int(ack.split(',')[1])
        print "acked number of packets is " +str(ackNumPackets)

        if ackNumPackets == numPackets:
            flag,modelsSent = sendModel(modelID,brokenBytes,modelsSent,conn)
        else:
            print "failed to send model: "+str(modelID)
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
        lenofModel = int(data.split(',')[3])

        conn.sendall('ACK,'+str(numPackets))

        return receiveData(sourceID,modelID,numPackets,lenofModel,conn)
        
    return 0

def receiveData(sourceID,modelID,numPackets,lenofModel,conn):
    # send ACK
    pickledModel = ""
    while (len(pickledModel)<lenofModel):
        #for i in range(0,numPackets):
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

def initiate(threadID,name,uid,PORT,fp,inFile,outFile,sFrom,sTo,weightType,recievedModels,runID,nums,cullThresh,miThresh):
    #print outFile
    #out = open(outFile,"a")
    out = open(os.devnull,'w')
    modelsSent = []
    args = ['python',fp,str(threadID),str(PORT),str(sFrom),str(sTo),inFile,str(INIT_DAYS),str(MODEL_HIST_THRESHOLD_ACC), 
            str(MODEL_HIST_THRESHOLD_PROB),str(STABLE_SIZE),str(MAX_WINDOW),str(THRESHOLD),str(weightType),str(runID),
            str(nums),str(uid),str(cullThresh),str(miThresh)]
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
        #else: recieveData(source_ID,s,conn,addr)
        
    p.wait()
    conn.close()
    s.close()
    out.close()


def getFPdict():
    FPdict = {
            'D001J001': '../../FollowingDistanceData/dr001J001.csv',
            'D001J002': '../../FollowingDistanceData/dr001J002.csv',
            'D001J003': '../../FollowingDistanceData/dr001J003.csv',
            'D002J001': '../../FollowingDistanceData/dr002J001.csv',
            'D002J002': '../../FollowingDistanceData/dr002J002.csv',
            'D002J003': '../../FollowingDistanceData/dr002J003.csv'}
    return FPdict

def main():
    global MODELS
    sourceInfo = dict()
    targetInfo = dict()
    runID = int(sys.argv[1])
    numStreams = int(sys.argv[2])
    socketOffset = int(sys.argv[3])
    weightType = str(sys.argv[4])
    CThresh = float(sys.argv[5])
    MThresh = float(sys.argv[6])

    FPdict = dict()
    FPdict = getFPdict()

    journeyList = ['D001J001','D001J002','D001J003','D002J001','D002J002','D002J003']#,

    random.shuffle(journeyList)
    journeyList = journeyList[0:numStreams]

    for idx, i in enumerate(journeyList):
        source = dict()
        source['Name'] = "source"+str(idx)+":"+str(i)
        source['uid'] = str(i)
        source['stdo']="TestResultsLog/Run"+str(runID)+"/"+str(source['Name'])+str(numStreams)+"Out.txt"
        source['PORT'] = socketOffset+idx
        source['Run'] = "source.py"
        source['stdin'] = FPdict[i]#"../../../TempModel/TestBedSim/Synthetic/"+str(FILEstdin[i])
        # source['weightType'] = 'OLS'
        source['weightType'] = weightType
        source['cullThresh'] = CThresh
        source['miThresh'] = MThresh

        sourceModels = dict()
        MODELS[idx] = sourceModels
        receivedModels = []
        sourceInfo[idx] = source

    print "creating threads"
    totalTime = 0
    for k,v in sourceInfo.iteritems():
        print k, v
        print "making thread"
        sThread = myThread(k,v,receivedModels,runID,numStreams)
        print "starting thread"
        sThread.start()
        totalTime = 0
        while not MODELS[k]:
            tts = random.uniform(0.0,1.2)
            time.sleep(10)
            totalTime = totalTime+ 100
            if totalTime >= 300:
                print " no stable models in :" +str(k)
                break
        print " RECIEVED FIRST MODEL SO STARTING NEXT THREAD"
    

if __name__ == '__main__':main()

