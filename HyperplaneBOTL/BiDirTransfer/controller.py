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
import sys

MODELS = dict()
INIT_DAYS = 30
STABLE_SIZE = 60
MAX_WINDOW = 30
MODEL_HIST_THRESHOLD_PROB = 0.1
MODEL_HIST_THRESHOLD_ACC = 0.55
THRESHOLD = 0.52


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
        print("starting "+ self.name)
        initiate(self.threadID,self.name,self.PORT,self.fp,self.inputFile,self.outputFile,self.sFrom,self.sTo,self.weightType,self.receivedModels)
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
        pickledModel = pickledModel + (conn.recv(1024))
    conn.sendall(('RECEIVED,'+str(modelID)).encode())

    storeModel(sourceID,modelID,pickledModel)
    return 1

def storeModel(sourceID,modelID,pickledModel):
    global MODELS
    model = pickle.loads(pickledModel)
    print(sourceID, modelID)
    MODELS[sourceID][modelID] = model

    print(sourceID, MODELS[sourceID])


def initiate(threadID,name,PORT,fp,inFile,outFile,sFrom,sTo,weightType,recievedModels):
    #print outFile
    out = open(outFile,"a")
    modelsSent = []
    args = ['python3',fp,str(threadID),str(PORT),str(sFrom),str(sTo),inFile,str(INIT_DAYS),str(MODEL_HIST_THRESHOLD_ACC), 
            str(MODEL_HIST_THRESHOLD_PROB),str(STABLE_SIZE),str(MAX_WINDOW),str(THRESHOLD),str(weightType)]
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

def getFPs(ident,driftType,offset):
    s = dict()
    FPs = dict()
    tos = dict()
    froms = dict()
    for i in range(1,6):
        s[(i+offset)]='TARGET'+str(i)+str(driftType)+str(ident+1)
        FPs[(i+offset)]='../../HyperplaneDG/Data/Datastreams/TARGET'+str(i)+'MultiConcept'+str(driftType)+str(ident+1)+'.csv'
        froms[(i+offset)]=1
        if driftType == 'Sudden':
            tos[(i+offset)]=10000
        else:
            tos[(i+offset)]=11900
    return s, FPs, froms, tos

def main():
    global MODELS
    sourceInfo = dict()
    targetInfo = dict()
    ident = int(sys.argv[1])
    driftType = str(sys.argv[2])
    offset = int(sys.argv[3])
    weightType = str(sys.argv[4])

    s, FPs, froms, tos = getFPs(ident,driftType,offset)
    for i,val in FPs.items():
        source = dict()
        source['Name'] = "source"+str(i)
        source['stdo']="./Out/multipleSudden"+str(i)+"Out.txt"
        source['PORT'] = i
        source['Run'] = "source.py"
        source['stdin'] = FPs[i]#"../../../TempModel/TestBedSim/Synthetic/"+str(FILEstdin[i])
        source['sFrom'] = 1
        source['sTo'] = 10000
        source['weightType'] = weightType 

        sourceModels = dict()
        MODELS[i] = sourceModels
        receivedModels = []
        sourceInfo[i] = source

    print("creating threads")

    for k,v in sourceInfo.items():
        print(k, v)
        print("making thread")
        sThread = myThread(k,v,receivedModels)
        print("starting thread")
        sThread.start()
        totalTime = 0
        while not MODELS[k]:
            time.sleep(10)
            # totalTime = totalTime+ 100
            # if totalTime >= 300:
                # print " no stable models in :" +str(k)
                # break
            print("waiting")
            #time.sleep(5)
    

if __name__ == '__main__':main()

