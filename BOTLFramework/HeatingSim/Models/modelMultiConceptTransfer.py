from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression as OLS


def calcWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType):
    if weightType =='OLS':
        return calcOLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSFE':
        return calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType == 'R2':
        return calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    print 'not the right metric'
    return 0


def calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    totalR2 = 0
    for k,v in sourceModels.iteritems():
        sourceP[k] = np.round(sourceModels[k].predict(X),decimals=1)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
        totalR2 += sourceR2[k]
            
    targetP = np.round(targetModel.predict(X),decimals=1)
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
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
    metaX['target'] = np.round(targetModel.predict(X),decimals=1)
    metaModel = OLS()
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
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= 0:
            dropKeys.append(k)
    metaX['target'] = np.round(targetModel.predict(X),decimals=1)
    
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

def updateInitialWeights(df,sourceModels,tLabel,DROP_FIELDS,weightType):
    if weightType =='OLS':
        return updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='OLSFE':
        return updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType == 'R2':
        return updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS)
    print 'not the right metric'
    return 0

def updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    for k,v in sourceModels.iteritems():
        sourceP[k] = np.round(sourceModels[k].predict(X),decimals=1)
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
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns

    return sourceOLS

def updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    dropKeys = []
    for k,v in sourceModels.iteritems():
        pred = np.round(sourceModels[k].predict(X),decimals=1)
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

def defaultInstancePredict(df,idx):
    df.loc[idx,'predictions'] = df.loc[idx,'oTemp']
    return df.loc[idx]

def instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    if weightType == 'OLS' or weightType =='OLSFE':
        metaX = pd.DataFrame(columns=weights['metaXColumns'])
    for k,v in sourceModels.iteritems():
        sourceP = np.round(sourceModels[k].predict(X),decimals=1)
        if weightType == 'OLS' or weightType == 'OLSFE':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights['sourceR2s'][k]/weights['totalR2']
            combo += weight*sourceP

    targetP = np.round(targetModel.predict(X),decimals=1)
    if weightType == 'OLS' or weightType =='OLSFE':
        metaX['target'] = targetP
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
    else:
        combo += (weights['targetR2']/weights['totalR2'])*targetP
    
    combo = np.round(combo,decimals=1)
    df.loc[idx,'predictions'] = combo
    
    return df.loc[idx]

def initialInstancePredict(df,idx,sourceModels,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    if weightType == 'OLS' or weightType =='OLSFE':
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
        sourceP = np.round(sourceModels[k].predict(X),decimals=1)
        if weightType == 'OLS' or weightType == 'OLSFE':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights[k]/sum(weights.values())
            combo += weight*sourceP
    
    if weightType == 'OLS' or weightType =='OLSFE':
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
    combo = np.round(combo,decimals=1)
    
    df.loc[idx,'predictions'] = combo
    return df.loc[idx]

def initialPredict(df,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    for idx in df.index:
        df.loc[idx] = instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
    return df