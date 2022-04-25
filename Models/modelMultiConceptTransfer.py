
import pandas as pd
import numpy as np
import operator
from sklearn import metrics
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import RidgeCV 
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import nnls
from Models.stsc import STSC

CULLTHRESH = 0
MITHRESH = 0
GAMMA = 0.1
BETA = 0.8

def calcWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,METASTATS,weightType,ct,mi,stableLocals=None,modelID=None,
        PCs=None,distanceMatrix=None,affinityMatrix=None,newTarget=False,groupedNames=None,metaModelKeep=None,recluster=False):
    global CULLTHRESH 
    global MITHRESH
    CULLTHRESH = ct
    MITHRESH = mi
    if weightType =='R2':
        return calcR2Weights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='OLS' or weightType == 'OLSPARed':
        return calcOLSWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='Ridge':
        return calcRidgeWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='NNLS':
        return calcNNLSWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='OLSFE' or weightType == 'OLSFEPA':
        return calcOLSFEWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS,distanceMatrix)
    # elif weightType =='OLSFEMI' or weightType == 'OLSFEMIPARed' or weightType == 'OLSFEMIRed':
    elif weightType =='OLSFEMI' or weightType == 'OLSFEMIPARed' or weightType == 'OLSFEMIRed':
        return calcOLSFEMIWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='OLSCL':
        return calcOLSCLWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='OLSCL2':
        return calcOLSCL2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,None,
                distanceMatrix,affinityMatrix,newTarget,groupedNames)
    elif weightType =='OLSPAC' or weightType == 'OLSKPAC':
        return calcOLSPACWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
                distanceMatrix,affinityMatrix,newTarget,groupedNames)
    elif weightType =='OLSPAC2' or weightType == 'OLSKPAC2' or weightType =='OLSKPAC2Red':
        return calcOLSCDOEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
                distanceMatrix,affinityMatrix,newTarget,groupedNames,metaModelKeep,recluster)
        # return calcOLSPAC2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID,PCs,
                # distanceMatrix,affinityMatrix,newTarget,groupedNames)
    # elif weightType == 'OLSAddExpA':
        # return calcOLSAddExpAWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID,PCs,
                # distanceMatrix,affinityMatrix,newTarget,groupedNames)
    # elif weightType == 'OLSAddExpP':
        # return calcOLSAddExpPWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID,PCs,
                # distanceMatrix,affinityMatrix,newTarget,groupedNames)
    elif weightType == 'OLSAWE':
        return calcOLSAWEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
                distanceMatrix,affinityMatrix,newTarget,groupedNames)
    else:
        return calcMSEWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS)



def calcR2Weights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    tc = 0 
    
    sourceP = dict()
    sourceR2 = dict()
    totalR2 = 0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        tc +=1
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
        totalR2 += sourceR2[k]
            
    targetP = targetModel.predict(X)
    targetR2 = metrics.r2_score(Y,targetP)
    tc +=1
    METASTATS['COMPSTATS']['R2Calcs'] +=1
    if targetR2 <= 0: 
        targetR2 = 0.00000000000001
    totalR2 += targetR2
    weights = {'sourceR2s':sourceR2,'targetR2':targetR2, 'totalR2':totalR2}
    return weights,tc,METASTATS

def calcOLSWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    tc=1
    
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    
    metaX[modelID] = targetModel.predict(X)
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns,'coeffs':metaModel.coef_}
    return weights,None,None,None,1,METASTATS

def calcNNLSWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    
    metaX[modelID] = targetModel.predict(X)
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)#.as_martix())
    
    sourceOLS = dict()
    weights = {'sourceR2s':sourceOLS,'targetR2':0, 'totalR2':0, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    return weights,1,METASTATS

def calcRidgeWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaX[modelID] = targetModel.predict(X)
    metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)#fit_intercept=False,tol=0.0001,solver='lsqr')
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    return weights,1,METASTATS

def calcOLSFEWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS,distanceMatrix):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    tc=0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        tc +=1
        if r2 <= CULLTHRESH:
            dropKeys.append(k)
    metaX[modelID] = targetModel.predict(X)
    
    if len(dropKeys) >0:
        metaX = metaX.drop(dropKeys,axis=1)
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights,distanceMatrix,None,None,tc,METASTATS

def calcOLSCLWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaX[modelID] = targetModel.predict(X)
    # print(metaX.columns)
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(metaX.columns))
    if len(metaX.columns) >1:
        # groupedNames,distanceMatrix,existingBases = STSC(metaX,'euclid','target',4,None,None) 
        # groupedNames,distanceMatrix,affinityMatrix = STSC(metaX,'euclid',modelID,4,None,None,True)
        k = int(len(metaX.coulmns)/2)
        k = 7
        groupedNames,distanceMatrix,affinityMatrix,tcPA,METASTATS = STSC(metaX,'euclid',modelID,k,METASTATS,None,None,True)
        METASTATS['COMPSTATS']['Clustering']+=1
        clusteredNames = [x for x in groupedNames if modelID in x][0]
        metaX = metaX[metaX.columns[metaX.columns.isin(clusteredNames)]]
        print("initial list: "+str(metaX.columns))
        print("cluster names: "+str(clusteredNames))
    print("number of models used: "+str(len(metaX.columns)))    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights,None,None,None,1,METASTATS#distanceMatrix,affinityMatrix,groupedNames

# def calcOLSCL2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID):
    # X = df.drop(DROP_FIELDS,axis=1).copy()
    # X = X.drop(tLabel,axis=1)
    # Y = df[tLabel].copy()
    # metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    # R2dict = dict()
    # dropKeys = []
    # for k,v in sourceModels.items():
        # pred = sourceModels[k]['model'].predict(X)
        # R2dict[k] = metrics.r2_score(Y,pred)
        # metaX[k] = pred
    # if stableLocals:
        # for k,v in stableLocals.items():
            # if k != modelID:
                # pred = stableLocals[k]['model'].predict(X)
                # R2dict[k] = metrics.r2_score(Y,pred)
                # metaX[k] = pred
    # targetPred = targetModel.predict(X)
    # metaX['target'] = targetPred
    # R2dict['target'] = metrics.r2_score(Y,targetPred)
    # # print(metaX.columns)
    # if len(metaX.columns) >1:
        # # groupedNames,distanceMatrix,existingBases = STSC(metaX,'euclid','target',4,None,None) 
        # k = int(len(metaX.coulmns)/2)
        # k=7
        # # groupedNames,distanceMatrix,affinityMatrix = STSC(metaX,'euclid',modelID,4,None,None,True)
        # groupedNames,distanceMatrix,affinityMatrix = STSC(metaX,'euclid',modelID,k,None,None,True)
        # metaModelKeep = ['target']
        # for c in groupedNames:
            # print("c in groupedNames")
            # print(c)
            # if 'target' in c :
                # c.remove('target')
            # if c:
                # maxGroupKey = max(c,key=R2dict.get)
                # metaModelKeep.append(maxGroupKey)
        # metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        # print("initial list: "+str(R2dict.keys()))
        # print("cluster names: "+str(metaX.columns))
    # print("number of models used: "+str(len(metaX.columns)))    
    # metaModel = OLS()
    # metaModel.fit(metaX,Y)
    
    # sourceOLS = dict()
    # for coef, feat in zip(metaModel.coef_,metaX.columns):
        # sourceOLS[feat]=coef
    # for k in dropKeys:
        # sourceOLS[k] = 0
    # targetOLS = sourceOLS['target']
    # del sourceOLS['target']
    # totalOLS = targetOLS + sum(sourceOLS.values())
    # weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    # return weights,None,None,None#distanceMatrix,affinityMatrix,groupedNames

def calcOLSCL2Weights(df,modelSet,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
        distanceMatrix=None,affinityMatrix=None,newTarget=False,groupedNames=None):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(modelSet.keys()))
    R2dict = dict()
    predsDict = dict()
    dropKeys = []
    tc=0
    tcPA=0
    # modelPCs = dict()
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(modelSet)+1)
    for k,v in modelSet.items():
        pred = modelSet[k]['model'].predict(X)
        R2dict[k] = metrics.r2_score(Y,pred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        metaX[k] = pred
        predsDict[k] = pred
        # modelPCs[k] = modelSet[k]['PCs']
    
    # modelPCs[modelID] = PCs
    targetPred = targetModel.predict(X)
    metaX[modelID] = targetPred
    R2dict[modelID] = metrics.r2_score(Y,targetPred)
    METASTATS['COMPSTATS']['R2Calcs'] +=1
    predsDict[modelID] = targetPred
    
    if len(predsDict)>1 and ((groupedNames is None) or newTarget):
        # groupedNames,distanceMatrix,existingBases = STSC(modelPCs,'principalAngles','target',4,distanceMatrix,existingBases)
        # groupedNames,distanceMatrix,affinityMatrix = STSC(modelPCs,'principalAngles',modelID,
                # 7,distanceMatrix,affinityMatrix,newTarget)
        # k = int(len(modelPCs)/2)
        k=7
        groupedNames,distanceMatrix,affinityMatrix,tcPA,METASTATS = STSC(predsDict,'euclid',modelID,
                k,METASTATS,distanceMatrix,affinityMatrix,newTarget)
        METASTATS['COMPSTATS']['Clustering']+=1
        
    if groupedNames:
        metaModelKeep = [modelID]
        for c in groupedNames:
            print("this c in groupedNames")
            print(c)
            compare = c.copy()
            if modelID in compare:
                compare.remove(modelID)
            tc+=len(compare)
            if compare:
                maxGroupKey = max(compare,key=R2dict.get)
                metaModelKeep.append(maxGroupKey)
        metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        print("initial list: "+str(R2dict.keys()))
        print("cluster names: "+str(metaX.columns))
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights,distanceMatrix,affinityMatrix,groupedNames,(tc,tcPA),METASTATS

def calcOLSPACWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
        distanceMatrix=None,affinityMatrix=None,newTarget=False,groupedNames=None):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list([modelID]))
    dropKeys = []
    metaX[modelID] = targetModel.predict(X)
    tcPA=0
    totalPAcalcs =0
    tc=0

    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    modelPCs = dict()
    for k,v in sourceModels.items():
        modelPCs[k] = sourceModels[k]['PCs']
    # print(stableLocals)
    if stableLocals and (modelID in stableLocals):
        modelPCs[modelID] = stableLocals[modelID]['PCs']
    else:
        modelPCs[modelID] = PCs
    if len(modelPCs)>1:
        # groupedNames,distanceMatrix,existingBases = STSC(modelPCs,'principalAngles','target',4,None,None)
        # groupedNames,distanceMatrix,affinityMatrix = STSC(modelPCs,'principalAngles',modelID,
                # 4,distanceMatrix,affinityMatrix,newTarget)
        k = int(len(modelPCs)/2)
        k=7
        groupedNames,distanceMatrix,affinityMatrix,tcPA,METASTATS = STSC(modelPCs,'principalAngles',modelID,
                k,METASTATS,distanceMatrix,affinityMatrix,newTarget)
        METASTATS['COMPSTATS']['Clustering']+=1
        totalPAcalcs+=tcPA
        clusteredNames = [x for x in groupedNames if modelID in x][0]
        # metaX = metaX[metaX.columns[metaX.columns.isin(clusteredNames)]]
        for i in clusteredNames:
            if i in sourceModels:
                metaX[i] = sourceModels[i]['model'].predict(X)
        print("initial list: "+str(sourceModels.keys()))
        print("cluster names: "+str(clusteredNames))


    print("number of models used: "+str(len(metaX.columns)))    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights,distanceMatrix,affinityMatrix,groupedNames,(0,totalPAcalcs),METASTATS

def calcNewAddExpWeight(df,idx,tLabel,DROP_FIELDS,METASTATS,modelSet,modelID,newModel,weights):
    idx = idx-1
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    mse = dict()
    weightedLoss = 0
    print(weights)
    print("gere")
    print(idx)
    metaModel = weights['metaModel']

    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(modelSet)+1)
    for k,v in modelSet.items():
        print(k)
        print(modelSet)
        if k != modelID and modelSet[k]['delAddExp']==0:
            pred = modelSet[k]['model'].predict(X)[0]
            print(pred)
            weightedLoss += metaModel[k]*(abs(pred-Y))
    pred = newModel.predict(X)[0]
    weightedLoss += (abs(pred-Y))
            
    print("weightedLoss:"+str(weightedLoss))
    # if len(metaModel.keys()) == 0:#1 and modelID in metaModel.keys():
        # pred = modelSet[k]['model'].predict(X)[0]
        # weightedLoss += (abs(pred-Y))
        # # metaModel[modelID] = 1
    # else: 
    metaModel[modelID] = GAMMA*weightedLoss
    print(metaModel)
    sourceOLS = dict()
    metaX = pd.DataFrame(columns = list(metaModel.keys()))
    for x in metaX.columns:
        sourceOLS[x]=metaModel[x]
    # for coef, feat in zip(metaModel.coef_,metaX.columns):
        # sourceOLS[feat]=coef
    # for k in dropKeys:
        # sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    print(metaModel)
    print("heree2222222")
    print("current modelID: "+str(modelID))
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns,'AWEmodelSet':modelSet}
    return weights,1,METASTATS#,None,None,None#distanceMatrix,affinityMatrix,groupedNames
    

def calcUpdateAddExpWeight(df,idx,modelSet,metaModel,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    idx = idx-1
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    print("updating")
    print(metaModel)
    print(modelID)
    print(modelSet)
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(modelSet)+1)
    for k,v in metaModel.items():
        if k != modelID:
            pred = modelSet[k]['model'].predict(X)
        else:
            pred = targetModel.predict(X)
        loss = abs(pred-Y)[0]
        print(loss)
        print(metaModel[k])
        metaModel[k] = metaModel[k]*(BETA**(loss))
        print(metaModel)
    sourceOLS = dict()
    metaX = pd.DataFrame(columns = list(metaModel.keys()))
    for x in metaX.columns:
        sourceOLS[x]=metaModel[x]
    # for coef, feat in zip(metaModel.coef_,metaX.columns):
        # sourceOLS[feat]=coef
    # for k in dropKeys:
        # sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns,'AWEmodelSet':modelSet}
    return weights,1,METASTATS#,None,None,None#distanceMatrix,affinityMatrix,groupedNames

# def calcOLSAddExpWeights(df,idx,modelSet,targetModel,tLabel,DROP_FIELDS,modelID,weights):
    # X = df.loc[idx].drop(DROP_FIELDS).copy()
    # X = X.drop(tLabel).values.reshape(1,-1)
    # Y = df.loc[idx,tLabel].copy()
    # mse = dict()
    # weightedLoss = 0
    # metaModel = weights['metaModel']
    # for k,v in modelSet.items():
        # if modelSet[k]['delAddExp']==1 and k in metaModel.keys():
            # del metaModel[k]
            # weights['metaXColumns'].remove(k)

    # if modelID not in metaModel.keys():
        # for k,v in modelSet.items():
            # if k != modelID and modelSet[k]['delAddExp'] == 0:
                # pred = modelSet[k]['model'].predict(X)
                # weightedLoss += metaModel[k]*(abs(pred-Y))
            
        # metaModel[modelID] = GAMMA*weightedLoss
    # else:
        # for k,v in modelSet.items():
            # pred = sourceModels[k].predict(X)
            # loss = abs(pred-Y)
            # # print(loss)
            # metaModel[k] = metaModel[k]*(BETA**(loss))

    # return metaModel,oldWeights


    # X = df.drop(DROP_FIELDS,axis=1).copy()
    # X = X.drop(tLabel,axis=1)
    # Y = df[tLabel].copy()
    # metaX = pd.DataFrame(columns = list(modelSet.keys()))
    # mse = dict()
    # metaModel = dict()
    # toDel = []
    # reducedModelSet = dict()
    # print(modelSet)
    # print(modelID)
    # for k,v in modelSet.items():
        # if modelSet[k]['delAWE'] == 0:
            # pred = modelSet[k]['model'].predict(X)
            # r2 = metrics.r2_score(Y,pred)
            # if r2 <=0:
                # print("deleting model"+str(k))
                # print(r2)
                # toDel.append(k)
                # modelSet[k]['delAWE'] = 1
            # else:
                # mse[k] = metrics.mean_squared_error(Y,pred)
                # reducedModelSet[k] = modelSet[k]
    # mse[modelID]=metrics.mean_squared_error(Y,targetModel.predict(X))
    # reducedModelSet[modelID] = {'model':targetModel}#dict()

        # # else: 
            # # toDel.append(k)
    # # for d in toDel:
        # # del modelSet[d]

    # totalMSE = sum(mse.values())
    # print("totalmse is:")
    # print(totalMSE)

    # for k,v in reducedModelSet.items():
        # metaModel[k] = (1 - (mse[k]/totalMSE))#/len(sourceModels)
        # # metaModel[k] = ((totalMSE/len(sourceModels)) - (mse[k]/totalMSE))#/len(sourceModels)
        # print("model:"+str(k))
        # print(str(metaModel[k])+": "+str(mse[k]))
    
    # # metaModel = OLS()
    # # metaModel.fit(metaX,Y)
    

def calcOLSAWEWeights(df,modelSet,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
        distanceMatrix=None,affinityMatrix=None,newTarget=False,groupedNames=None):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(modelSet.keys()))
    mse = dict()
    metaModel = dict()
    toDel = []
    reducedModelSet = dict()
    tc=0
    print(modelSet)
    print(modelID)
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(modelSet)+1)
    for k,v in modelSet.items():
        if modelSet[k]['delAWE'] == 0:
            print(modelSet[k])
            pred = modelSet[k]['model'].predict(X)
            r2 = metrics.r2_score(Y,pred)
            METASTATS['COMPSTATS']['R2Calcs'] +=1
            tc +=1
            if r2 <=0:
                print("deleting model"+str(k))
                print(r2)
                toDel.append(k)
                modelSet[k]['delAWE'] = 1
            else:
                mse[k] = metrics.mean_squared_error(Y,pred)
                reducedModelSet[k] = modelSet[k]
    mse[modelID]=metrics.mean_squared_error(Y,targetModel.predict(X))
    reducedModelSet[modelID] = {'model':targetModel}#dict()

        # else: 
            # toDel.append(k)
    # for d in toDel:
        # del modelSet[d]

    totalMSE = sum(mse.values())
    print("totalmse is:")
    print(totalMSE)

    for k,v in reducedModelSet.items():
        metaModel[k] = (1 - (mse[k]/totalMSE))#/len(sourceModels)
        # metaModel[k] = ((totalMSE/len(sourceModels)) - (mse[k]/totalMSE))#/len(sourceModels)
        print("model:"+str(k))
        print(str(metaModel[k])+": "+str(mse[k]))
    
    # metaModel = OLS()
    # metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    metaX = pd.DataFrame(columns = list(reducedModelSet.keys()))
    for x in metaX.columns:
        sourceOLS[x]=metaModel[k]
    # for coef, feat in zip(metaModel.coef_,metaX.columns):
        # sourceOLS[feat]=coef
    # for k in dropKeys:
        # sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns,'AWEmodelSet':modelSet}
    return weights,None,None,None,tc,METASTATS#distanceMatrix,affinityMatrix,groupedNames


def calcOLSCDOEWeights(df,modelSet,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
        distanceMatrix=None,affinityMatrix=None,newTarget=False,groupedNames=None,metaModelKeep=None,recluster=False):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(modelSet.keys()))
    R2dict = dict()
    dropKeys = []
    modelPCs = dict()
    totalCalcs=0
    tc = 0
    tcPA=0
    totalPAcalcs =0
    print("modelset: "+str(modelSet.keys()))
    if distanceMatrix and affinityMatrix:
        print("distanceMatrix:"+str(distanceMatrix.keys()))
        print("affinityMatrix:"+str(affinityMatrix.keys()))
    
    if len(modelSet.keys())>1 and ((groupedNames is None) or newTarget or recluster):
        print("doing clustering")
        # groupedNames,distanceMatrix,existingBases = STSC(modelPCs,'principalAngles','target',4,distanceMatrix,existingBases)
        # groupedNames,distanceMatrix,affinityMatrix = STSC(modelPCs,'principalAngles',modelID,
                # 7,distanceMatrix,affinityMatrix,newTarget)
        METASTATS['evalModelSet']+=1
        METASTATS['sizeModelSet'].append(len(modelSet)+1)
        for k,v in modelSet.items():
            pred = modelSet[k]['model'].predict(X)
            R2dict[k] = metrics.r2_score(Y,pred)
            METASTATS['COMPSTATS']['R2Calcs'] +=1
            tc+=1
            metaX[k] = pred
            modelPCs[k] = modelSet[k]['PCs']
        
        if 'temp' not in str(modelID):
            modelPCs[modelID] = PCs
        targetPred = targetModel.predict(X)
        metaX[modelID] = targetPred
        R2dict[modelID] = metrics.r2_score(Y,targetPred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        tc+=1
        k = int(len(modelPCs)/2)
        k=7
        groupedNames,distanceMatrix,affinityMatrix,tcPA,METASTATS = STSC(modelPCs,'principalAngles',modelID,
                k,METASTATS,distanceMatrix,affinityMatrix,newTarget)
        METASTATS['COMPSTATS']['Clustering']+=1
        totalPAcalcs+=tcPA
        
        if groupedNames:
            metaModelKeep = [modelID]
            for c in groupedNames:
                print("c in groupedNames")
                print(c)
                compare = c.copy()
                if modelID in compare:
                    compare.remove(modelID)
                # tc+=len(c)
                if compare:
                    maxGroupKey = max(compare,key=R2dict.get)
                    metaModelKeep.append(maxGroupKey)
            metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
            print("initial list: "+str(R2dict.keys()))
            print("cluster names: "+str(metaX.columns))
    else:
        print("not clustering, metaX keys are:")
        print(metaModelKeep)
        print("but grouped names are")
        print(groupedNames)
        print("and model set is: "+str(modelSet.keys()))
        print("stableLocals")
        print(stableLocals)

        toKeep = []
        for k,mod in enumerate(metaModelKeep):
            if stableLocals:
                if (mod in stableLocals): 
                    print(k,mod)
                    toKeep.append(metaModelKeep[k])
            if modelSet:
                if (mod in modelSet) and mod not in toKeep:
                    print(k,mod)
                    toKeep.append(metaModelKeep[k])
        metaModelKeep = toKeep
        metaX = pd.DataFrame(columns = metaModelKeep)
        for m in metaModelKeep:#k,v in modelSet.items():
            # if m == modelID:
                # pred = targetModel.predict(X)
                # metaX[m] = pred
            if m in modelSet.keys():
                pred = modelSet[m]['model'].predict(X)
                metaX[m] = pred
            # modelPCs[k] = modelSet[k]['PCs']
    # if modelID not in metaX.columns():
        # pred = targetModel.predict(X)
        # metaX[m] = pred


    modelPCs[modelID] = PCs
    targetPred = targetModel.predict(X)
    metaX[modelID] = targetPred
    if metaX.isnull().values.any():
        print("META X HAS NULLS")
        print(metaX)
        print(modelSet)
        print(distanceMatrix.keys())
        print(metaModelKeep)
    # R2dict[modelID] = metrics.r2_score(Y,targetPred)
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    print("grouped names at end")
    print(groupedNames)
    print("metaX columns")
    print(metaX.columns)
    print("number of instances used to metalearn: "+str(len(metaX)))
    return weights,distanceMatrix,affinityMatrix,groupedNames,(tc,totalPAcalcs),METASTATS


def calcOLSPAC2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,METASTATS,stableLocals,modelID,PCs,
        distanceMatrix=None,affinityMatrix=None,newTarget=False,groupedNames=None):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    R2dict = dict()
    dropKeys = []
    modelPCs = dict()
    tc=0
    tcPA=0
    totalPAcalcs =0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+len(stableLocals)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        R2dict[k] = metrics.r2_score(Y,pred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        metaX[k] = pred
        modelPCs[k] = sourceModels[k]['PCs']
    if stableLocals:
        for k,v in stableLocals.items():
            if k != modelID:
                pred = stableLocals[k]['model'].predict(X)
                R2dict[k] = metrics.r2_score(Y,pred)
                METASTATS['COMPSTATS']['R2Calcs'] +=1
                metaX[k] = pred
                modelPCs[k] = stableLocals[k]['PCs']
    if stableLocals and (modelID in stableLocals):
        modelPCs[modelID] = stableLocals[modelID]['PCs']
    else:
        modelPCs[modelID] = PCs
    targetPred = targetModel.predict(X)
    metaX[modelID] = targetPred
    R2dict[modelID] = metrics.r2_score(Y,targetPred)
    METASTATS['COMPSTATS']['R2Calcs'] +=1
    tc +=1
    # print(metaX.columns)
    if len(modelPCs)>1 and ((groupedNames is None) or newTarget):
        print("about to do STSC modelID is "+str(modelID))
        if stableLocals and (modelID in stableLocals):
            modelPCs[modelID] = stableLocals[modelID]['PCs']
        for k,v in modelPCs.items():
            print("key:"+str(k)+" shape:"+str(v.shape))

        # groupedNames,distanceMatrix,existingBases = STSC(modelPCs,'principalAngles','target',4,distanceMatrix,existingBases)
        # groupedNames,distanceMatrix,affinityMatrix = STSC(modelPCs,'principalAngles',modelID,
                # 4,distanceMatrix,affinityMatrix,newTarget)
        k = int(len(modelPCs)/2)
        k=7
        groupedNames,distanceMatrix,affinityMatrix,tcPA,METASTATS = STSC(modelPCs,'principalAngles',modelID,
                k,METASTATS,distanceMatrix,affinityMatrix,newTarget)
        METASTATS['COMPSTATS']['Clustering']+=1
        totalPAcalcs+=tcPA
        metaModelKeep = [modelID]
        for c in groupedNames:
            print("c in groupedNames")
            print(c)
            if modelID in c :
                c.remove(modelID)
            if modelID in c:
                c.remove(modelID)
            tc+=len(c)
            if c:
                maxGroupKey = max(c,key=R2dict.get)
                metaModelKeep.append(maxGroupKey)
        metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        print("initial list: "+str(R2dict.keys()))
        print("cluster names: "+str(metaX.columns))

    if groupedNames:
        metaModelKeep = [modelID]
        for c in groupedNames:
            print("c in groupedNames")
            print(c)
            if modelID in c :
                c.remove(modelID)
            if modelID in c:
                c.remove(modelID)
            tc+=len(c)
            if c:
                maxGroupKey = max(c,key=R2dict.get)
                metaModelKeep.append(maxGroupKey)
        metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        print("initial list: "+str(R2dict.keys()))
        print("cluster names: "+str(metaX.columns))
        
    print("number of models used: "+str(len(metaX.columns)))    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights,distanceMatrix,affinityMatrix,groupedNames,(tc,totalPAcalcs),METASTATS


def calcOLSFEMIWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    tc=0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        if 'toDiscard' in v.keys() and v['toDiscard']== True and k != modelID:
            dropKeys.append(k)
    for k,v in sourceModels.items():
        if k not in dropKeys:
            pred = sourceModels[k]['model'].predict(X)
            metaX[k] = pred
            r2 = metrics.r2_score(Y,pred)
            METASTATS['COMPSTATS']['R2Calcs'] +=1
            tc +=1
            # print(str(k)+": "+str(r2))
            if r2 <= CULLTHRESH and k != modelID:
                dropKeys.append(k)
    
    metaX[modelID] = targetModel.predict(X)
    
    for k,v in sourceModels.items():
        if k in dropKeys or k == modelID:
            continue
        for l,u in sourceModels.items():
            if (l in dropKeys) or l==k or l == modelID:
                continue
            predA = sourceModels[k]['model'].predict(X)
            predB = sourceModels[l]['model'].predict(X)
            # combinedpredA = np.stack((predA,np.ones(len(predA)))).T
            # mi = mutual_info_regression(combinedpredA,predB)[0]
            predDFs=pd.DataFrame(dict(predA=predA,predB=predB))
            mi = mutual_info_regression(predDFs,predB)
            METASTATS['COMPSTATS']['MICalcs']+=1
            mi=mi[0]/mi[1]
            # print("mis "+str(k)+str(l)+": "+str(mi))
            tc +=1
            if mi > MITHRESH:
                r2A = metrics.r2_score(Y,predA)
                r2B = metrics.r2_score(Y,predB)
                METASTATS['COMPSTATS']['R2Calcs'] +=2
                tc +=2
                if r2A>r2B:
                    dropKeys.append(l)
                else:
                    dropKeys.append(k)

    if len(dropKeys) >0:
        print("dropping: "+str(dropKeys))
        metaX = metaX.drop(dropKeys,axis=1)
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS[modelID]
    del sourceOLS[modelID]
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights,None,None,None,tc,METASTATS

def calcMSEWeights(df,sourceModels,targetModel,modelID,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    tc=0
    sourceP = dict()
    sourceMSE = dict()
    totalMSE = 0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceMSE[k] =  metrics.mean_squared_error(Y,sourceP[k])
        tc +=1
        if sourceMSE[k] <= 0:
            sourceMSE[k] = 0.00000000000001
        sourceMSE[k] = 1/sourceMSE[k]
        totalMSE += sourceMSE[k]
            
    targetP = targetModel.predict(X)
    targetMSE = metrics.mean_squared_error(Y,targetP)
    tc +=1
    if targetMSE <= 0: 
        targetMSE = 0.00000000000001
    totalMSE += targetMSE
    weights = {'sourceR2s':sourceMSE,'targetR2':targetMSE, 'totalR2':totalMSE}
    return weights,tc,METASTATS

def updateInitialWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS,weightType,ct,mi,distanceMatrix = None, 
        affinityMatrix = None, groupedNames=None):
    global CULLTHRESH
    global MITHRESH
    CULLTHRESH = ct
    MITHRESH = mi
    if weightType == 'R2':
        return updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='OLS':
        return updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='Ridge':
        return updateInitialRidgeWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='NNLS':
        return updateInitialNNLSWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS)
    elif weightType =='OLSFE' or weightType == 'OLSFEPA':
        return updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS,distanceMatrix)
    # elif weightType =='OLSFEMI' or weightType == 'OLSFEMIPARed':
    elif weightType =='OLSFEMI' or weightType == 'OLSFEMIPARed' or weightType == 'OLSFEMIRed':
        return updateInitialOLSFEMIWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS)
    elif 'OLSCL' in weightType:
        return updateInitialOLSCL2Weights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS,distanceMatrix,affinityMatrix,groupedNames)
    elif weightType == 'MSE':
        return updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS)
    elif 'PAC' in weightType:
        return updateInitialOLSPACWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS,distanceMatrix,affinityMatrix,groupedNames)
    else:
        return updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS,distanceMatrix)

def updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    tc=0
    
    sourceP = dict()
    sourceR2 = dict()
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        tc +=1
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
    
    return (sourceR2,tc),METASTATS

def updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    
    return (sourceOLS,1),METASTATS

def updateInitialRidgeWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)#fit_intercept=False,tol=0.0001,solver='lsqr')
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    
    return (sourceOLS,1),METASTATS

def updateInitialNNLSWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)#.as_martix())
    
    sourceOLS = dict()
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    return (sourceOLS,1),METASTATS

def updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS,distanceMatrix):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = dict()
    tc=0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        tc +=1
        if r2 <= CULLTHRESH:
            dropKeys[k] = r2
    if len(list(dropKeys.keys())) == len(metaX.columns):
        for i in range(0,3):
            if len(list(dropKeys.keys()))>0:
                maxKey = max(iter(dropKeys.items()),key=operator.itemgetter(1))[0]
                del dropKeys[maxKey]
    
    if len(list(dropKeys.keys())) > 0 and len(list(dropKeys.keys())) < len(metaX.columns):
        metaX = metaX.drop(list(dropKeys.keys()),axis=1)
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    return (sourceOLS,tc),METASTATS

def updateInitialOLSFEMIWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = dict()
    tc=0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        tc +=1
        if r2 <= CULLTHRESH:
            dropKeys[k] = r2
    
    for k,v in sourceModels.items():
        if k in list(dropKeys.keys()):
            continue
        for l,u in sourceModels.items():
            if (l in list(dropKeys.keys())) or l==k:
                continue
            predA = sourceModels[k]['model'].predict(X)
            predB = sourceModels[l]['model'].predict(X)
            tc +=1
            # combinedpredA = np.stack((predA,np.ones(len(predA)))).T
            # mi = mutual_info_regression(combinedpredA,predB)[0]
            predDFs=pd.DataFrame(dict(predA=predA,predB=predB))
            mi = mutual_info_regression(predDFs,predB)
            METASTATS['COMPSTATS']['MICalcs']+=1
            mi=mi[0]/mi[1]
            if mi > MITHRESH:
                r2A = metrics.r2_score(Y,predA)
                r2B = metrics.r2_score(Y,predB)
                # METASTATS['COMPSTATS']['R2Calcs'] +=2
                # tc +=2
                if r2A>r2B:
                    dropKeys[l] = r2B
                else:
                    dropKeys[k] = r2A

    if len(list(dropKeys.keys())) == len(metaX.columns):
        for i in range(0,3):
            if len(list(dropKeys.keys()))>0:
                maxKey = max(iter(dropKeys.items()),key=operator.itemgetter(1))[0]
                del dropKeys[maxKey]
    
    if len(list(dropKeys.keys())) > 0 and len(list(dropKeys.keys())) < len(metaX.columns):
        metaX = metaX.drop(list(dropKeys.keys()),axis=1)

    metaModel = OLS()
    metaModel.fit(metaX,Y)
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    print("sourceOLS is: "+str(sourceOLS))
    return (sourceOLS,tc),METASTATS

def updateInitialOLSCL2Weights(df,modelSet,tLabel,DROP_FIELDS,METASTATS,
        distanceMatrix=None,affinityMatrix=None,newTarget=False,groupedNames=None):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    tc=0
    metaX = pd.DataFrame(columns = list(modelSet.keys()))
    R2dict = dict()
    dropKeys = []
    predsDict = dict()
    tcPA=0
    totalPAcalcs =0
    # modelPCs = dict()
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(modelSet)+1)
    for k,v in modelSet.items():
        pred = modelSet[k]['model'].predict(X)
        R2dict[k] = metrics.r2_score(Y,pred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        tc +=1
        metaX[k] = pred
        predsDict[k] = pred
        # modelPCs[k] = modelSet[k]['PCs']
    
    # modelPCs[modelID] = PCs
    # targetPred = targetModel.predict(X)
    # metaX[modelID] = targetPred
    # R2dict[modelID] = metrics.r2_score(Y,targetPred)
    if groupedNames is None:
        newTarget=True
    else:
        newTarget = False
    k = 7
    if len(predsDict)>1:
        groupedNames,distanceMatrix,affinityMatrix,tcPA,METASTATS = STSC(predsDict,'euclid',None,
                k,METASTATS,distanceMatrix,affinityMatrix,newTarget)
        METASTATS['COMPSTATS']['Clustering']+=1
        totalPAcalcs+=tcPA

    if groupedNames:
        metaModelKeep = []#'target']
        for c in groupedNames:
            print("c in groupedNames")
            print(c)
            # if 'target' in c :
                # c.remove('target')
            # if modelID in c:
                # c.remove(modelID)
            if c:
                maxGroupKey = max(c,key=R2dict.get)
                metaModelKeep.append(maxGroupKey)
        metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        print("initial list: "+str(R2dict.keys()))
        print("cluster names: "+str(metaX.columns))
    else:
        metaModelKeep = predsDict.keys()
        metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
    
    # if len(predsDict)>1 and ((groupedNames is None) or newTarget):
        # # groupedNames,distanceMatrix,existingBases = STSC(modelPCs,'principalAngles','target',4,distanceMatrix,existingBases)
        # # groupedNames,distanceMatrix,affinityMatrix = STSC(modelPCs,'principalAngles',modelID,
                # # 7,distanceMatrix,affinityMatrix,newTarget)
        # # k = int(len(modelPCs)/2)
        # k=7
        # groupedNames,distanceMatrix,affinityMatrix = STSC(predsDict,'euclid',None,
                # k,distanceMatrix,affinityMatrix,newTarget)
        
    # if groupedNames:
        # metaModelKeep = []#modelID]
        # for c in groupedNames:
            # print("c in groupedNames")
            # print(c)
            # if modelID in c:
                # c.remove(modelID)
            # if c:
                # maxGroupKey = max(c,key=R2dict.get)
                # metaModelKeep.append(maxGroupKey)
        # metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        # print("initial list: "+str(R2dict.keys()))
        # print("cluster names: "+str(metaX.columns))
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    print("sourceOLS is: "+str(sourceOLS))
    # targetOLS = sourceOLS['target']
    # del sourceOLS['target']
    # totalOLS = targetOLS + sum(sourceOLS.values())
    # weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return (sourceOLS,distanceMatrix,affinityMatrix,groupedNames,(tc,totalPAcalcs)),METASTATS
    

def updateInitialOLSPACWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS,distanceMatrix=None,
        affinityMatrix=None,groupedNames=None):
    print(DROP_FIELDS)
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    tc=0
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    R2dict = dict()
    dropKeys = []
    modelPCs = dict()
    tcPA=0
    totalPAcalcs =0
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        R2dict[k] = metrics.r2_score(Y,pred)
        METASTATS['COMPSTATS']['R2Calcs'] +=1
        metaX[k] = pred
        modelPCs[k] = sourceModels[k]['PCs']
    # if stableLocals:
        # for k,v in stableLocals.items():
            # if k != modelID:
                # pred = stableLocals[k]['model'].predict(X)
                # R2dict[k] = metrics.r2_score(Y,pred)
                # metaX[k] = pred
                # modelPCs[k] = stableLocals[k]['PCs']
    # if stableLocals and (modelID in stableLocals):
        # modelPCs['target'] = stableLocals[modelID]['PCs']
    # else:
        # modelPCs['target'] = PCs
    # targetPred = targetModel.predict(X)
    # metaX['target'] = targetPred
    # R2dict['target'] = metrics.r2_score(Y,targetPred)
    # # print(metaX.columns)
    # if len(modelPCs)>1 and ((groupedNames is None) or newTarget):
        # print("about to do STSC modelID is "+str(modelID))
        # if stableLocals and (modelID in stableLocals):
            # modelPCs[modelID] = stableLocals[modelID]['PCs']
        # for k,v in modelPCs.items():
            # print("key:"+str(k)+" shape:"+str(v.shape))

        # groupedNames,distanceMatrix,existingBases = STSC(modelPCs,'principalAngles','target',4,distanceMatrix,existingBases)
        # groupedNames,distanceMatrix,affinityMatrix = STSC(modelPCs,'principalAngles',modelID,
                # 4,distanceMatrix,affinityMatrix,newTarget)
    k = int(len(modelPCs)/2)
    k=7
    if groupedNames is None:
        newTarget=True
    else:
        newTarget = False

    if len(modelPCs)>1:
        groupedNames,distanceMatrix,affinityMatrix,tcPA,METASTATS = STSC(modelPCs,'principalAngles',None,
                k,METASTATS,distanceMatrix,affinityMatrix,newTarget)
        METASTATS['COMPSTATS']['Clustering']+=1
        totalPAcalcs+=tcPA

    if groupedNames:
        metaModelKeep = []#'target']
        for c in groupedNames:
            print("c in groupedNames")
            print(c)
            # if 'target' in c :
                # c.remove('target')
            # if modelID in c:
                # c.remove(modelID)
            tc +=len(c)
            if c:
                maxGroupKey = max(c,key=R2dict.get)
                metaModelKeep.append(maxGroupKey)
        metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        print("initial list: "+str(R2dict.keys()))
        print("cluster names: "+str(metaX.columns))
    else:
        metaModelKeep = modelPCs.keys()
        metaX = metaX[metaX.columns[metaX.columns.isin(metaModelKeep)]]
        
    print("number of models used: "+str(len(metaX.columns)))    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    print("sourceOLS is: "+str(sourceOLS))
    # targetOLS = sourceOLS['target']
    # del sourceOLS['target']
    # totalOLS = targetOLS + sum(sourceOLS.values())
    # weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return (sourceOLS,distanceMatrix,affinityMatrix,groupedNames,(tc,totalPAcalcs)),METASTATS

def updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS,METASTATS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    tc=0
    sourceP = dict()
    sourceMSE = dict()
    METASTATS['evalModelSet']+=1
    METASTATS['sizeModelSet'].append(len(sourceModels)+1)
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceMSE[k] = metrics.mean_squared_error(Y,sourceP[k])
        tc +=1
        if sourceMSE[k] <= 0:
            sourceMSE[k] = 0.00000000000001
        sourceMSE[k] = 1/sourceMSE[k]
    return (sourceMSE,0),METASTATS

def defaultInstancePredict(df,idx,DEFAULT_PRED):
    if DEFAULT_PRED == 'Following':
        df.loc[idx,'predictions']=3.0
    elif DEFAULT_PRED == 'Heating':
        df.loc[idx,'predictions'] = df.loc[idx,'oTemp']
    else:
        df.loc[idx,'predictions']=0.5
    return df.loc[idx],0,[]

def instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType,
        stableLocals = None,modelID = None,PCs = None):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    # if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'OLSFEMI' or weightType == 'OLSCL' or weightType == 'Ridge' or weightType == 'NNLS':
    if 'AddExp' in weightType:
        pred = 0
        metaModel = weights['metaModel']
        print("metaModel: "+str(metaModel))
        totalmetaModel = sum(metaModel.values())
        # metaX = pd.DataFrame(columns=weights['metaXColumns'])
        # print(totalmetaModel)
        # if len(metaX.columns)>1:
        for k in metaModel.keys():
            if len(metaModel.keys()) == 1:
                pred = metaModel[k]*targetModel.predict(X)

            elif len(sourceModels)>0:
                if k == modelID:
                    # print("here: metaModel val and total")
                    # print(metaModel[k])
                    # print(totalmetaModel)
                    temppred = metaModel[k]*targetModel.predict(X)
                    print("modid and temppred")
                    print(str(k)+": "+str(temppred))
                    pred += temppred
                else:
                    pred += metaModel[k]*sourceModels[k]['model'].predict(X)
            else:
                pred = targetModel.predict(X)
        # print("metaModel: "+str(metaModel))
        # print("total weights: "+str(totalmetaModel))
        # print("pred is: "+str(pred))
        if totalmetaModel != 0:
            print("pred:"+str(pred))
            print("totalmetaModel:"+str(totalmetaModel))
            print("metamodel:" +str(metaModel))
            df.loc[idx,'predictions'] = pred/totalmetaModel
        else:
            df.loc[idx,'predictions'] = targetModel.predict(X)
        print("actual: "+str(df.loc[idx,tLabel]))
        print("overall pred:" +str(df.loc[idx,'predictions']))
        bases = []
        basesSize = 0
        for y in metaModel.keys():
            bases.append(y)
            basesSize+=1
        if modelID is not None and modelID not in bases:
            bases.append(modelID)
            basesSize+=1
        return df.loc[idx],basesSize,bases


    if 'AWE' in weightType:
        metaModel = weights['metaModel']
        totalmetaModel = sum(metaModel.values())
        # print(totalmetaModel)
        pred = 0
        metaX = pd.DataFrame(columns=weights['metaXColumns'])
        if len(metaX.columns)>1:
            for k in metaX.columns:
                if k == modelID:
                    # print("here: metaModel val and total")
                    # print(metaModel[k])
                    # print(totalmetaModel)
                    pred += (metaModel[k]/totalmetaModel)*targetModel.predict(X)
                else:
                    pred += (metaModel[k]/totalmetaModel)*sourceModels[k]['model'].predict(X)
        else:
            pred = targetModel.predict(X)

        # for k,v in sourceModels.items():
            # if k in metaX.columns:
        # for k,v in sourceModels.items():
            # if len(sourceModels)>1:
                # if sourceModels[k]['delAWE'] == 0:
                    # pred += (metaModel[k]/totalmetaModel)*sourceModels[k]['model'].predict(X)
            # else:
                # pred = sourceModels[k]['model'].predict(X)
        df.loc[idx,'predictions'] = pred
        bases = []
        basesSize = 0
        for y in metaX.columns:
            bases.append(y)
            basesSize+=1
        if modelID is not None and modelID not in bases:
            bases.append(modelID)
            basesSize+=1
        return df.loc[idx],basesSize,bases

    if weightType == 'OLSCL2' or 'OLSPAC' in weightType or 'OLSKPAC' in weightType:
        metaX = pd.DataFrame(columns=weights['metaXColumns'])
        print(str(idx)+": metaX columns should be: "+str(metaX.columns))
        for k,v in sourceModels.items():
            if k in metaX.columns:
                metaX[k] = sourceModels[k]['model'].predict(X)
        if stableLocals:
            print("stableLocals:"+str(stableLocals.keys()))
            for k,v in stableLocals.items():
                if k in metaX.columns and k != modelID and k != modelID:
                    metaX[k] = stableLocals[k]['model'].predict(X)
        # if metaX.isnull().values.any():
            # print("META X HAS NULLS")
            # print(metaX)
            # print(modelSet)
            # print(distanceMatrix.keys())
            # print(metaModelKeep)
    else:
        if 'OLS' in weightType or weightType == 'Ridge' or weightType == 'NNLS':
            metaX = pd.DataFrame(columns=weights['metaXColumns'])
            print("metaX columns are: "+str(weights['metaXColumns']))
            print(sourceModels.keys())
        for k,v in sourceModels.items():
            sourceP = sourceModels[k]['model'].predict(X)
            # if weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'OLSFEMI' or weightType == 'OLSCL' or weightType == 'Ridge' or weightType =='NNLS':
            if 'OLS' in weightType or weightType == 'Ridge' or weightType == 'NNLS':
                if k in metaX.columns:
                    metaX[k] = sourceP
            else:
                weight = weights['sourceR2s'][k]/weights['totalR2']
                combo += weight*sourceP
    
    targetP = targetModel.predict(X)
    # if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'OLSFEMI' or weightType == 'OLSCL' or weightType == 'Ridge':
    if 'OLS' in weightType or weightType == 'Ridge':
        metaX[modelID] = targetP
        metaModel = weights['metaModel']
        print("metaX:"+str(metaX))
        print("currentModel is: "+str(modelID))
        if metaX.isnull().values.any():
            print("META X HAS NULLS")
            print(metaX)
            print(distanceMatrix.keys())
            print(metaModelKeep)
            print(modelSet.keys())
        
        print("using: "+str(weights['sourceR2s']))
        print(metaModel)
        combo = metaModel.predict(metaX)
        df.loc[idx,'predictions'] = combo
        print(str(idx)+": prediction is: "+str(combo))
        # print("using: "+str(weights['sourceR2s']))
    elif weightType =='NNLS':
        metaX[modelID] = targetP
        metaModel = weights['metaModel']
        combo = 0
        for w in range(0,metaModel.size):
            if metaModel.size == 1:
                combo += metaModel.item(0) * metaX.as_matrix().item(0,0)
            else:
                combo += metaModel.item(w) * metaX.as_matrix().item((0,w))
        df.loc[idx,'predictions'] = combo
    else:
        combo += (weights['targetR2']/weights['totalR2'])*targetP
        df.loc[idx,'predictions'] = combo
    bases = []
    basesSize = 0
    for y in metaX.columns:
        bases.append(y)
        basesSize+=1
    if modelID is not None and modelID not in bases:
        bases.append(modelID)
        basesSize+=1
    
    return df.loc[idx],basesSize,bases

def getEvenWeights(df,sourceModels,tLabel,DROP_FIELDS):
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    sourceOLS = dict()
    for feat in metaX.columns:
        sourceOLS[feat] = 1/len(metaX.columns)
    return sourceOLS,metaX

def initialInstancePredict(df,idx,sourceModels,weights,tLabel,DROP_FIELDS,METASTATS,weightType,ct,mi,distanceMatrix=None,
        affinityMatrix=None,groupedNames=None):
    global CULLTHRESH
    global MITHRESH
    CULLTHRESH = ct
    MITHRESH = mi
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    metaX = pd.DataFrame()
    tc=0
    # if weightType == 'OLS' or weightType == 'Ridge' or weightType =='OLSFE' or weightType == 'NNLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
    if 'OLS' in weightType or weightType == 'Ridge' or weightType == 'NNLS':
        print("weights:"+str(weights))
        if 'metaXColumns' not in weights:
            historyData = df.loc[:idx].copy()
            historyData = historyData.drop(idx,axis=0)
            if len(historyData)<20:
                weightType = 'Even'
                weights,metaX = getEvenWeights(historyData,sourceModels,tLabel,DROP_FIELDS)
            else:
                if 'PAC' in weightType or 'CL' in weightType:
                    (weights,distanceMatrix,affinityMatrix,groupedNames,tc),METASTATS = updateInitialWeights(historyData,sourceModels,
                            tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,MITHRESH,distanceMatrix,affinityMatrix,groupedNames)
                else:
                    (weights,tc),METASTATS = updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,METASTATS,weightType,CULLTHRESH,
                            MITHRESH)

        if not weightType =='Even':
            metaX = pd.DataFrame(columns=weights['metaXColumns'])
    
    for k,v in sourceModels.items():
        sourceP = sourceModels[k]['model'].predict(X)
        # if weightType == 'OLS' or weightType == 'Ridge' or weightType == 'OLSFE' or weightType == 'NNLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
        if 'OLS' in weightType or weightType == 'Ridge' or weightType == 'NNLS':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights[k]/sum(weights.values())
            combo += weight*sourceP

    
    # if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'Ridge' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
    if 'OLS' in weightType or weightType == 'Ridge':
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
    elif weightType =='NNLS':
        metaModel = weights['metaModel']
        combo = 0
        for w in range(0,metaModel.size):
            if metaModel.size == 1:
                combo += metaModel.item(0) * metaX.as_matrix().item(0,0)
            else:
                combo += metaModel.item(w) * metaX.as_matrix().item((0,w))
    
    df.loc[idx,'predictions'] = combo
    bases = []
    basesSize = 0
    for y in metaX.columns:
        bases.append(y)
        basesSize+=1
    # if modelID is not None and modelID not in bases:
        # bases.append(modelID)
        # basesSize+=1
    if 'PAC' in weightType or 'CL' in weightType:
        # print(df.loc[idx],distanceMatrix,affinityMatrix,groupedNames)
        return df.loc[idx],distanceMatrix,affinityMatrix,groupedNames,basesSize,bases,tc,METASTATS
    return df.loc[idx],basesSize,bases,tc,METASTATS

def initialPredict(df,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    for idx in df.index:
        df.loc[idx],w = instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
    return df
