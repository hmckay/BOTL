from __future__ import division
import numpy as np
import pandas as pd
import random
import math
from optparse import OptionParser


def getRandomComponents(num_attrs,num_select):
    attr_list = []
    for i in range(0,num_select):
        attr_list.append(random.randrange(0,num_attrs,1))
    return attr_list

def getFunctionList(funcs,num_concepts,func_names):
    remainingFuncs = funcs
    funcList = []
    while remainingFuncs >= len(func_names):
        funcList.extend(func_names)
        remainingFuncs -= len(func_names)
    
    if remainingFuncs < len(func_names):
        funcList.extend(random.sample(func_names,remainingFuncs))
    random.shuffle(funcList)
    #fs = funcList

    fullRepeat = int(num_concepts/len(funcList))
    remainderRepeat = num_concepts%len(funcList)
    funcsToUse = []
    for i in range(0,fullRepeat):
        funcsToUse.extend(funcList)
    if remainderRepeat > 0:
        funcsToUse.extend(random.sample(funcList,remainderRepeat))
    print funcsToUse
    random.shuffle(funcsToUse)
    return funcsToUse


def writeToFile(f,data):
    for i in data:
        for index,j in enumerate(i):
            if index == len(i)-1:
                f.write(str(j)+"\n")
            else:
                f.write(str(j)+",")

def func1(attrs,features):
    print "func1: "+str(attrs)
    tot = 0
    for i in attrs:
        tot += features[i]
    return tot/len(attrs)

def func2(attrs,features):
    print "func2: "+str(attrs)
    no_attrs = len(attrs)
    tot = 0
    for i in attrs:
        tot += math.cos(features[i])
    return tot/no_attrs

def func3(attrs,features):
    print "func3: "+str(attrs)
    no_attrs = len(attrs)
    tot = 0
    for i in attrs:
        tot += math.sin(features[i])
    return tot/no_attrs

def func4(attrs,features):
    print "func4: "+str(attrs)
    tot = 0
    for i in attrs:
        tot += features[i]**2
    return math.sqrt(tot)

def func5(attrs,features):
    if len(set(attrs)) == 1:
        return func1(attrs,features)
    
    print "func5: "+str(attrs)
    tot = 0
    for i in attrs:
        tot = abs(tot-features[i])
    return tot

def shuffleComponents(comp,funcs,startID):
    print startID
    shuffComp = dict()
    shuffFunc = dict()
    ids = comp.keys()
    shuffIds = ids
    shuffIds.remove(startID)
    shuffIds = random.sample(shuffIds,len(shuffIds))
    print ids
    print shuffIds
    shuffComp[startID] = comp[startID]
    shuffFunc[startID] = funcs[startID]

    for idx,i in enumerate(shuffIds):
        shuffComp[idx+1] = comp[i]
        shuffFunc[idx+1] = funcs[i]
    print comp
    print shuffComp

    return shuffComp, shuffFunc


def addGradualDrift(driftLen,index,schedules,conceptComponents,conceptLengths,conflicting,NOISE_PROB,num_attrs,functions,f):
    concept1Len = conceptLengths[index]
    concept2Len = conceptLengths[index+1]
    i = schedules[index]
    j = schedules[index+1]
    concept1Comp = conceptComponents[i]
    concept2Comp = conceptComponents[j]
    #driftLen = 5
    dataset = []
    print "driftlen is: "+str(driftLen)
    for n in range(0,driftLen):
        concept1Prob = (driftLen-n)/driftLen
        print concept1Prob
        
        features = []
        y=0
        for x in range(0,num_attrs):
            features.append(random.random())
        a = []
        
        if random.random() <= concept1Prob:
            for xi in concept1Comp:
                a.append(xi)
            y = functions[i](a,features)
        else:
            for xi in concept2Comp:
                a.append(xi)
            y = functions[j](a,features)
        
        if random.random() < NOISE_PROB:
            y += 0.05

        if y < 0 : y = 0
        elif y > 1: y = 1
        features.append(y)
        dataset.append(features)
        #print dataset
    writeToFile(f,dataset)
    return 0




def createDataset(NUM_CONCEPTS,functions,num_attrs,DEPENDANT,filename,NOISE_PROB,schedules,conceptLengths,conflicting,conceptComponents,combinedC,combinedKeys,gradual,driftLen):
    f = open(filename,'a')
    if conflicting:
        f.write("#Number of concepts: "+str(NUM_CONCEPTS*2)+"\n")
    else:
        f.write("#Number of concepts: "+str(NUM_CONCEPTS)+"\n")
    f.write("#Number of attributes the concept is dependent on: "+str(DEPENDANT)+"\n")
    f.write("#Concept components: "+str(conceptComponents)+"\n")
    f.write("#Noise prob: "+str(NOISE_PROB)+"\n")
    f.write("#Order of concepts: "+str(schedules)+"\n")
    f.write("#Length of concepts: "+str(conceptLengths)+"\n")
    f.write("#Conflicting concepts: "+str(conflicting)+"\n")
    f.write("#Functions used: "+str(functions)+"\n")
    f.write("#Composite concepts: "+ str(combinedKeys)+"\n")
    f.write("#Gradual drift: "+ str(gradual)+"\n")
    f.write("#Gradual drift length: "+str(driftLen)+"\n")
    
    for xi in range(1,num_attrs+1):
        f.write(str(xi)+",")
    f.write("y\n")



    print conceptComponents
    if not combinedC:
        for index,i in enumerate(schedules):
            conceptComp = conceptComponents[i]
            conceptLen = conceptLengths[index]
            dataset = []
            while len(dataset) < conceptLen:
        
                features = []
                y=0
                for x in range(0,num_attrs):
                    features.append(random.random())
                a = []
                for xi in conceptComp:
                    a.append(xi)
                y = functions[i](a,features)
                if conflicting and i-NUM_CONCEPTS>=0:
                    y = 1-y
                    print "conflicting concept: "+str(i)
                if random.random() < NOISE_PROB:
                    y += 0.05

                if y < 0 : y = 0
                elif y > 1: y = 1
                features.append(y)
                dataset.append(features)
            #print dataset
            writeToFile(f,dataset)
            if gradual and (index+1 < len(schedules)):
                addGradualDrift(driftLen,index,schedules,conceptComponents,conceptLengths,conflicting,NOISE_PROB,num_attrs,functions,f)

    if combinedC:
        print combinedKeys
        for index, i in enumerate(schedules):
            conceptComp = conceptComponents[i]
            conceptLen = conceptLengths[index]
            dataset = []
            print i
            while len(dataset) < conceptLen:
                features = []
                y=0
                for x in range(0,num_attrs):
                    features.append(random.random())
                
                if i in combinedKeys:
                    a = []
                    for j in combinedKeys:
                        print conceptComponents[j]
                        for xi in conceptComponents[j]:
                            a.append(xi)
                else:
                    a = []
                    for xi in conceptComp:
                        a.append(xi)
                y = functions[i](a,features)
                
                if random.random() < NOISE_PROB:
                    y+=0.05
                if y < 0: y = 0
                elif y> 1: y =1
                features.append(y)
                dataset.append(features)
            writeToFile(f,dataset)


    f.close()

def main():
    
    parser = OptionParser(usage="usage: prog options filename",version="prog 1.0")
    parser.add_option("-w","--filepath",default="tempSimulationData.csv",dest="OUT_FP",help="heating data output filepath")
    parser.add_option("-n","--noise",default=0.2,dest="NOISE_PROB",help="probability of noise")
    parser.add_option("-y","--concepts",default=2,dest="CONCEPTS",help="number of concepts")
    parser.add_option("-s","--schedules",default=0,dest="SCHEDULES",help="order of schedules used")
    parser.add_option("-l","--length",default=0,dest="LENGTH",help="length of each concept")
    parser.add_option("-d","--dependent",default=3,dest="DEPENDANT",help="number of attributes the concept is dependant upon")
    parser.add_option("-a","--attributes",default=5,dest="ATTRIBUTES",help="number of attributes")
    parser.add_option("-c","--conflicting",default=0,dest="CONFLICTING",help="are there conflicting concepts? (1=yes)")
    parser.add_option("-f","--functions",default=1, dest="FUNCTIONS",help="number of functions to use, default=1")
    parser.add_option("-t","--transfer",default=-1,dest="TRANSFER",help="number of concepts shared in SOURCE and TARGET, default=-1 meaning all concepts are shared")
    parser.add_option("-r","--randomOrder",default=0,dest="RANDORDER",help="do you want your concepts in a random order, default = 0 meaning no")
    parser.add_option("-b","--combinedConcept",default=0,dest="COMBINED",help="do you want a composite concept, default = 0 meaning no")
    parser.add_option("-g","--gradualDrift",default=0,dest="GRADUAL",help="type of drift. Gradual = 1, sudden = 0, default = 0")
    parser.add_option("-i","--driftLength",default=0,dest="DRIFTLEN",help="for gradual drift, the length for change over between two concepts")

    (options,args) = parser.parse_args()
    NOISE_PROB = float(options.NOISE_PROB)
    NUM_CONCEPTS = int(options.CONCEPTS)
    DEPENDANT = int(options.DEPENDANT)
    mask = options.SCHEDULES
    num_attrs = int(options.ATTRIBUTES)
    conflicting = int(options.CONFLICTING)
    funcs = int(options.FUNCTIONS)
    shared = int(options.TRANSFER)
    randOrder = int(options.RANDORDER)
    combinedC = int(options.COMBINED)
    gradual = int(options.GRADUAL)
    driftLen = int(options.DRIFTLEN)

    if shared == -1:
        shared = NUM_CONCEPTS
    schedules = []
    if mask:
        schedules = [int(i) for i in mask.split(',')]
    else: mask = 0
    strLengths = options.LENGTH
    conceptLengths = [int(i) for i in strLengths.split(',')]
    conceptLengths = [500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500]
    
    func_names = [func1]#,func2]#,func3]#,func4,func5]

    combinedKeys = []
    if combinedC:
        combinedKeys = random.sample(range(0,NUM_CONCEPTS),2)
        print combinedKeys

    

    if conflicting:
        conceptComponents = dict()
        functions = dict()
        NUM_CONCEPTS = NUM_CONCEPTS/2
        possibleFuncs = getFunctionList(funcs,NUM_CONCEPTS,func_names)
        print possibleFuncs
        for i in range(0,NUM_CONCEPTS):
            conceptComponents[i] = getRandomComponents(num_attrs,DEPENDANT)
            conceptComponents[NUM_CONCEPTS+i] = conceptComponents[i]
            functions[i] = possibleFuncs[i]
            functions[NUM_CONCEPTS+i] = functions[i]
        createDataset(NUM_CONCEPTS,functions,num_attrs,DEPENDANT,"SOURCE"+options.OUT_FP,NOISE_PROB,schedules,conceptLengths,conflicting,conceptComponents,0,combinedKeys,gradual,driftLen)
        print functions
        for i in range(0,NUM_CONCEPTS):
            conceptComponents[NUM_CONCEPTS+i] = conceptComponents[i]
            functions[NUM_CONCEPTS+i] = functions[i]
        print conceptComponents
        if randOrder:
            conceptComponents,functions = shuffleComponents(conceptComponents,functions,min(conceptComponents.keys()))
        createDataset(NUM_CONCEPTS*2,functions,num_attrs,DEPENDANT,"TARGET"+options.OUT_FP,NOISE_PROB,schedules,conceptLengths,0,conceptComponents,0,combinedKeys,gradual,driftLen)
    else:
        conceptComponents = dict()
        functions = dict()
        possibleFuncs = getFunctionList(funcs,NUM_CONCEPTS,func_names)
        print possibleFuncs
        for i in range(0,NUM_CONCEPTS):
            conceptComponents[i] = getRandomComponents(num_attrs,DEPENDANT)
            functions[i] = possibleFuncs[i]
        print functions
        createDataset(NUM_CONCEPTS,functions,num_attrs,DEPENDANT,"Data/SOURCE"+options.OUT_FP+".csv",NOISE_PROB,schedules,conceptLengths,conflicting,conceptComponents,0,combinedKeys,gradual,driftLen)
        for o in range(1,6):
            if shared > 0:
                
                change = random.sample(range(0,NUM_CONCEPTS),NUM_CONCEPTS-shared)
                for i in change:
                    if i == 0:
                        i = 1
                    conceptComponents[i] = getRandomComponents(num_attrs,DEPENDANT)
                    functions[i] = possibleFuncs[i]
                
                if randOrder:
                    conceptComponents,functions = shuffleComponents(conceptComponents,functions,min(conceptComponents.keys()))
                for k in range(1,6):
                    createDataset(NUM_CONCEPTS,functions,num_attrs,DEPENDANT,"Data/TARGET"+str(o)+options.OUT_FP+str(k)+".csv",NOISE_PROB,schedules,conceptLengths,conflicting,conceptComponents,combinedC,combinedKeys,gradual,driftLen)
            else:
                createDataset(NUM_CONCEPTS,functions,num_attrs,DEPENDANT,"TARGET"+str(o)+options.OUT_FP,NOISE_PROB,schedules,conceptLengths,conflicting,conceptComponents,0,combinedKeys,gradual,driftLen)

    

if __name__ == "__main__": main()
