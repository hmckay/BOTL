from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import RidgeCV 
from sklearn.linear_model import Ridge 
from scipy.optimize import nnls

def calcWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType):
    if weightType =='R2':
        return calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLS':
        return calcOLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='Ridge':
        return calcRidgeWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='NNLS':
        return calcNNLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSFE':
        return calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    else:
        return calcMSEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)



def calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    totalR2 = 0
    for k,v in sourceModels.iteritems():
        sourceP[k] = sourceModels[k].predict(X)
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
    metaX = pd.DataFrame(columns = sourceModels.keys())
    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    metaX['target'] = targetModel.predict(X)
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns,'coeffs':metaModel.coef_}
    return weights

def calcNNLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())

    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    
    metaX['target'] = targetModel.predict(X)
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)
    sourceOLS = dict()
    weights = {'sourceR2s':sourceOLS,'targetR2':0, 'totalR2':0, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    return weights

def calcRidgeWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    metaX['target'] = targetModel.predict(X)
    metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)
    metaModel.fit(metaX,Y)
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    return weights

def calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    dropKeys = []
    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= 0:
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
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    return weights

def calcMSEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceMSE = dict()
    totalMSE = 0
    for k,v in sourceModels.iteritems():
        sourceP[k] = sourceModels[k].predict(X)
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

def updateInitialWeights(df,sourceModels,tLabel,DROP_FIELDS,weightType):
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
    else:
        return updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS)

def updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    for k,v in sourceModels.iteritems():
        sourceP[k] = sourceModels[k].predict(X)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001

    return sourceR2

def updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())

    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
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
    metaX = pd.DataFrame(columns = sourceModels.keys())
    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)
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
    metaX = pd.DataFrame(columns = sourceModels.keys())
    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)
    
    sourceOLS = dict()
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    #print sourceOLS
    #print metaX
    return sourceOLS

def updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    dropKeys = []
    for k,v in sourceModels.iteritems():
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= 0:
            dropKeys.append(k)
    if len(dropKeys) > 0 and len(dropKeys) < len(metaX.columns):
        metaX = metaX.drop(dropKeys,axis=1)
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

def updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceMSE = dict()
    for k,v in sourceModels.iteritems():
        sourceP[k] = sourceModels[k].predict(X)
        sourceMSE[k] = metrics.mean_squared_error(Y,sourceP[k])
        if sourceMSE[k] <= 0:
            sourceMSE[k] = 0.00000000000001
        sourceMSE[k] = 1/sourceMSE[k]
    return sourceMSE

def defaultInstancePredict(df,idx):
    df.loc[idx,'predictions']=0.5
    return df.loc[idx]

def instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'Ridge' or weightType == 'NNLS':
        metaX = pd.DataFrame(columns=weights['metaXColumns'])
    for k,v in sourceModels.iteritems():
        sourceP = sourceModels[k].predict(X)
        if weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'Ridge' or weightType =='NNLS':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights['sourceR2s'][k]/weights['totalR2']
            combo += weight*sourceP

    targetP = targetModel.predict(X)
    if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'Ridge':
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
    
    return df.loc[idx]

def initialInstancePredict(df,idx,sourceModels,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    if weightType == 'OLS' or weightType == 'Ridge' or weightType =='OLSFE' or weightType == 'NNLS':
        if 'metaXColumns' not in weights:
            historyData = df.loc[:idx].copy()
            historyData = historyData.drop(idx,axis=0)
            if len(historyData)<1:
                weightType = 'R2'
            else:
                weights = updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,weightType)

        if not weightType =='R2':
            metaX = pd.DataFrame(columns=weights['metaXColumns'])
    for k,v in sourceModels.iteritems():
        print "k is: "+str(k)
        sourceP = sourceModels[k].predict(X)
        if weightType == 'OLS' or weightType == 'Ridge' or weightType == 'OLSFE' or weightType == 'NNLS':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights[k]/sum(weights.values())
            combo += weight*sourceP

    if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'Ridge':
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