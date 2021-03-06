
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

def calcWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType,ct,mi,stableLocals=None,modelID=None,PCs=None):
    global CULLTHRESH 
    global MI
    CULLTHRESH = ct
    MITHRESH = mi
    if weightType =='R2':
        return calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLS':
        print("OLS weights")
        return calcOLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='Ridge':
        return calcRidgeWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='NNLS':
        return calcNNLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSFE':
        return calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSFEMI':
        return calcOLSFEMIWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSCL':
        return calcOLSCLWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSCL2':
        return calcOLSCL2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID)
    elif weightType =='OLSPAC' or weightType == 'OLSKPAC':
        return calcOLSPACWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID,PCs)
    elif weightType =='OLSPAC2' or weightType == 'OLSKPAC2':
        return calcOLSPAC2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID,PCs)
    else:
        return calcMSEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)



def calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    totalR2 = 0
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
        totalR2 += sourceR2[k]
            
    targetP = targetModel.predict(X)
    targetR2 = metrics.r2_score(Y,targetP)
    if targetR2 <= 0: 
        targetR2 = 0.00000000000001
    totalR2 += targetR2
    weights = {'sourceR2s':sourceR2,'targetR2':targetR2, 'totalR2':totalR2}
    return weights

def calcOLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    
    metaX['target'] = targetModel.predict(X)
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns,'coeffs':metaModel.coef_}
    return weights

def calcNNLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    
    metaX['target'] = targetModel.predict(X)
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)#.as_martix())
    
    sourceOLS = dict()
    weights = {'sourceR2s':sourceOLS,'targetR2':0, 'totalR2':0, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    return weights

def calcRidgeWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaX['target'] = targetModel.predict(X)
    metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)#fit_intercept=False,tol=0.0001,solver='lsqr')
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    return weights

def calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= CULLTHRESH:
            dropKeys.append(k)
    metaX['target'] = targetModel.predict(X)
    
    if len(dropKeys) >0:
        metaX = metaX.drop(dropKeys,axis=1)
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights

def calcOLSCLWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaX['target'] = targetModel.predict(X)
    # print(metaX.columns)
    if len(metaX.columns) >1:
        groupedNames = STSC(metaX,'euclid','target',4) 
        clusteredNames = [x for x in groupedNames if 'target' in x][0]
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
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights

def calcOLSCL2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    R2dict = dict()
    dropKeys = []
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        R2dict[k] = metrics.r2_score(Y,pred)
        metaX[k] = pred
    if stableLocals:
        for k,v in stableLocals.items():
            if k != modelID:
                pred = stableLocals[k]['targetModel'].predict(X)
                R2dict[k] = metrics.r2_score(Y,pred)
                metaX[k] = pred
    targetPred = targetModel.predict(X)
    metaX['target'] = targetPred
    R2dict['target'] = metrics.r2_score(Y,targetPred)
    # print(metaX.columns)
    if len(metaX.columns) >1:
        groupedNames = STSC(metaX,'euclid','target',4) 
        metaModelKeep = ['target']
        for c in groupedNames:
            print("c in groupedNames")
            print(c)
            if 'target' in c :
                c.remove('target')
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
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights

def calcOLSPACWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID,PCs):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(['target']))
    dropKeys = []
    metaX['target'] = targetModel.predict(X)

    modelPCs = dict()
    for k,v in sourceModels.items():
        modelPCs[k] = sourceModels[k]['PCs']
    print(stableLocals)
    if stableLocals and (modelID in stableLocals):
        modelPCs['target'] = stableLocals[modelID]['PCs']
    else:
        modelPCs['target'] = PCs
    if len(modelPCs)>1:
        groupedNames = STSC(modelPCs,'principalAngles','target',4)
        clusteredNames = [x for x in groupedNames if 'target' in x][0]
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
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights

def calcOLSPAC2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,stableLocals,modelID,PCs):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    R2dict = dict()
    dropKeys = []
    modelPCs = dict()
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        R2dict[k] = metrics.r2_score(Y,pred)
        metaX[k] = pred
        modelPCs[k] = sourceModels[k]['PCs']
    if stableLocals:
        for k,v in stableLocals.items():
            if k != modelID:
                pred = stableLocals[k]['targetModel'].predict(X)
                R2dict[k] = metrics.r2_score(Y,pred)
                metaX[k] = pred
                modelPCs[k] = stableLocals[k]['PCs']
    if stableLocals and (modelID in stableLocals):
        modelPCs['target'] = stableLocals[modelID]['PCs']
    else:
        modelPCs['target'] = PCs
    targetPred = targetModel.predict(X)
    metaX['target'] = targetPred
    R2dict['target'] = metrics.r2_score(Y,targetPred)
    # print(metaX.columns)
    if len(modelPCs)>1:
        groupedNames = STSC(modelPCs,'principalAngles','target',4)
        metaModelKeep = ['target']
        for c in groupedNames:
            print("c in groupedNames")
            print(c)
            if 'target' in c :
                c.remove('target')
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
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights

def calcOLSFEMIWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= CULLTHRESH:
            dropKeys.append(k)
    
    metaX['target'] = targetModel.predict(X)
    
    for k,v in sourceModels.items():
        if k in dropKeys:
            continue
        for l,u in sourceModels.items():
            if (l in dropKeys) or l==k:
                continue
            predA = sourceModels[k]['model'].predict(X)
            predB = sourceModels[l]['model'].predict(X)
            combinedpredA = np.stack((predA,np.ones(len(predA)))).T
            mi = mutual_info_regression(combinedpredA,predB)[0]
            if mi > MITHRESH:
                r2A = metrics.r2_score(Y,predA)
                r2B = metrics.r2_score(Y,predB)
                if r2A>r2B:
                    dropKeys.append(l)
                else:
                    dropKeys.append(k)

    if len(dropKeys) >0:
        metaX = metaX.drop(dropKeys,axis=1)
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights

def calcMSEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceMSE = dict()
    totalMSE = 0
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceMSE[k] =  metrics.mean_squared_error(Y,sourceP[k])
        if sourceMSE[k] <= 0:
            sourceMSE[k] = 0.00000000000001
        sourceMSE[k] = 1/sourceMSE[k]
        totalMSE += sourceMSE[k]
            
    targetP = targetModel.predict(X)
    targetMSE = metrics.mean_squared_error(Y,targetP)
    if targetMSE <= 0: 
        targetMSE = 0.00000000000001
    totalMSE += targetMSE
    weights = {'sourceR2s':sourceMSE,'targetR2':targetMSE, 'totalR2':totalMSE}
    return weights

def updateInitialWeights(df,sourceModels,tLabel,DROP_FIELDS,weightType,ct,mi):
    global CULLTHRESH
    global MITHRESH
    CULLTHRESH = ct
    MITHRESH = mi
    if weightType == 'R2':
        return updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='OLS':
        return updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='Ridge':
        return updateInitialRidgeWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='NNLS':
        return updateInitialNNLSWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='OLSFE':
        return updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='OLSFEMI':
        return updateInitialOLSFEMIWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif 'OLSCL' in weightType or 'OLSPAC' in weightType or 'OLSKPAC' in weightType:
        return updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS)
    else:
        return updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS)

def updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
    
    return sourceR2

def updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
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
    
    return sourceOLS

def updateInitialRidgeWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    
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
    
    return sourceOLS

def updateInitialNNLSWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)#.as_martix())
    
    sourceOLS = dict()
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    return sourceOLS

def updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = dict()
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
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
    return sourceOLS

def updateInitialOLSFEMIWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = dict()
    for k,v in sourceModels.items():
        pred = sourceModels[k]['model'].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
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
            combinedpredA = np.stack((predA,np.ones(len(predA)))).T
            mi = mutual_info_regression(combinedpredA,predB)[0]
            if mi > MITHRESH:
                r2A = metrics.r2_score(Y,predA)
                r2B = metrics.r2_score(Y,predB)
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
    return sourceOLS

def updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceMSE = dict()
    for k,v in sourceModels.items():
        sourceP[k] = sourceModels[k]['model'].predict(X)
        sourceMSE[k] = metrics.mean_squared_error(Y,sourceP[k])
        if sourceMSE[k] <= 0:
            sourceMSE[k] = 0.00000000000001
        sourceMSE[k] = 1/sourceMSE[k]
    return sourceMSE

def defaultInstancePredict(df,idx):
    df.loc[idx,'predictions']=df.loc[idx,'oTemp']
    return df.loc[idx]

def instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType,
        stableLocals = None,modelID = None,PCs = None):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    # if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'OLSFEMI' or weightType == 'OLSCL' or weightType == 'Ridge' or weightType == 'NNLS':
    if weightType == 'OLSCL2' or 'OLSPAC' in weightType or 'OLSKPAC' in weightType:
        metaX = pd.DataFrame(columns=weights['metaXColumns'])
        for k,v in sourceModels.items():
            if k in metaX.columns:
                metaX[k] = sourceModels[k]['model'].predict(X)
        if stableLocals:
            for k,v in stableLocals.items():
                if k in metaX.columns and k != 'target' and k != modelID:
                    metaX[k] = stableLocals[k]['targetModel'].predict(X)
    else:
        if 'OLS' in weightType or weightType == 'Ridge' or weightType == 'NNLS':
            metaX = pd.DataFrame(columns=weights['metaXColumns'])
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
        metaX['target'] = targetP
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
        df.loc[idx,'predictions'] = combo
    elif weightType =='NNLS':
        metaX['target'] = targetP
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
    # print('BOTL instance predict: '+str(df.loc[idx,'predictions']))
    return df.loc[idx]

def getEvenWeights(df,sourceModels,tLabel,DROP_FIELDS):
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    sourceOLS = dict()
    for feat in metaX.columns:
        sourceOLS[feat] = 1/len(metaX.columns)
    return sourceOLS

def initialInstancePredict(df,idx,sourceModels,weights,tLabel,DROP_FIELDS,weightType,ct,mi):
    global CULLTHRESH
    global MITHRESH
    CULLTHRESH = ct
    MITHRESH = mi
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    # if weightType == 'OLS' or weightType == 'Ridge' or weightType =='OLSFE' or weightType == 'NNLS' or weightType == 'OLSFEMI' or weightType == 'OLSCL':
    if 'OLS' in weightType or weightType == 'Ridge' or weightType == 'NNLS':
        if 'metaXColumns' not in weights:
            historyData = df.loc[:idx].copy()
            historyData = historyData.drop(idx,axis=0)
            if len(historyData)<10:
                weightType = 'Even'
                weights = getEvenWeights(historyData,sourceModels,tLabel,DROP_FIELDS)
            else:
                weights = updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH)

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
    return df.loc[idx]

def initialPredict(df,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    for idx in df.index:
        df.loc[idx] = instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
    return df
