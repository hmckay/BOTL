
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression as OLS
import operator
from sklearn.feature_selection import mutual_info_regression

#DELTA = 0.025
#FORGETTING_FACTOR = 0.995
CULLTHRESH = 0.2
MITHRESH = 0.95

def calcWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS,weightType):
    if weightType =='OLS':
        return calcOLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSFE':
        return calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType =='OLSFEMI':
        return calcOLSFEMIWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    elif weightType == 'R2':
        return calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
    print('not the right metric')
    return 0


def calcR2Weights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    totalR2 = 0
    for k,v in sourceModels.items():
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
    #ws = (sourceR2/(sourceR2+targetR2))
    #wt = (targetR2/(sourceR2+targetR2))
    weights = {'sourceR2s':sourceR2,'targetR2':targetR2, 'totalR2':totalR2}
    return weights

def calcOLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    #print X
    for k,v in sourceModels.items():
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
    metaX['target'] = np.round(targetModel.predict(X),decimals=1)
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    #print model.coef_
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    #print sourceOLS
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    #print weights
    return weights

def calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    #print X
    for k,v in sourceModels.items():
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
    #print model.coef_
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    #print sourceOLS
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    #print weights
    return weights

def calcOLSFEMIWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = []
    #print X
    for k,v in sourceModels.items():
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= CULLTHRESH:
            dropKeys.append(k)
    metaX['target'] = np.round(targetModel.predict(X),decimals=1)
    
    for k,v in sourceModels.items():
        if k in dropKeys:
            continue
        for l,u in sourceModels.items():
            if (l in dropKeys) or l==k:
                continue
            predA = np.round(sourceModels[k].predict(X),decimals=1)
            predB = np.round(sourceModels[l].predict(X),decimals=1)
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
    #print model.coef_
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    #print sourceOLS
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.values())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    #print weights
    return weights

def updateInitialWeights(df,sourceModels,tLabel,DROP_FIELDS,weightType):
    if weightType =='OLS':
        return updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='OLSFE':
        return updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType =='OLSFEMI':
        return updateInitialOLSFEMIWeights(df,sourceModels,tLabel,DROP_FIELDS)
    elif weightType == 'R2':
        return updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS)
    print('not the right metric')
    return 0

def updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    for k,v in sourceModels.items():
        sourceP[k] = np.round(sourceModels[k].predict(X),decimals=1)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
    return sourceR2

def updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    #print X
    for k,v in sourceModels.items():
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
    #print metaX
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    #print model.coef_
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    #print sourceOLS
    #print metaX
    return sourceOLS


def updateInitialOLSFEWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = dict()
    for k,v in sourceModels.items():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= 0:
            dropKeys[k] = r2
    #print metaX
    if len(list(dropKeys.keys())) == len(metaX.columns):
        for i in range(0,3):
            if len(list(dropKeys.keys()))>0:
                maxKey = max(iter(dropKeys.items()),key=operator.itemgetter(1))[0]
                del dropKeys[maxKey]
    
    if len(list(dropKeys.keys())) > 0 and len(list(dropKeys.keys())) < len(metaX.columns):
        metaX = metaX.drop(list(dropKeys.keys()),axis=1)
    
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    #print model.coef_
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    #print sourceOLS
    #print metaX
    return sourceOLS

def updateInitialOLSFEMIWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    dropKeys = dict()
    for k,v in sourceModels.items():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = np.round(sourceModels[k].predict(X),decimals=1)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= CULLTHRESH:
            dropKeys[k] = r2
    #print metaX
    for k,v in sourceModels.items():
        if k in list(dropKeys.keys()):
            continue
        for l,u in sourceModels.items():
            if (l in list(dropKeys.keys())) or l==k:
                continue
            predA = np.round(sourceModels[k].predict(X),decimals=1)
            predB = np.round(sourceModels[l].predict(X),decimals=1)
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
    #print model.coef_
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    for k in dropKeys:
        sourceOLS[k] = 0
    sourceOLS['metaModel'] = metaModel
    sourceOLS['metaXColumns'] = metaX.columns
    #print sourceOLS
    #print metaX
    return sourceOLS


def defaultInstancePredict(df,idx):
    df.loc[idx,'predictions'] = df.loc[idx,'oTemp']
    return df.loc[idx]

def instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'OLSFEMI':
        metaX = pd.DataFrame(columns=weights['metaXColumns'])
    for k,v in sourceModels.items():
        sourceP = np.round(sourceModels[k].predict(X),decimals=1)
        if weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'OLSFEMI':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights['sourceR2s'][k]/weights['totalR2']
            combo += weight*sourceP

    targetP = np.round(targetModel.predict(X),decimals=1)
    if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'OLSFEMI':
        metaX['target'] = targetP
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
    else:
        combo += (weights['targetR2']/weights['totalR2'])*targetP
    
    combo = np.round(combo,decimals=1)
    df.loc[idx,'predictions'] = combo
    
    return df.loc[idx]

def getEvenWeights(df,sourceModels,tLabel,DROP_FIELDS):
    metaX = pd.DataFrame(columns = list(sourceModels.keys()))
    sourceOLS = dict()
    for feat in metaX.columns:
        sourceOLS[feat] = 1/len(metaX.columns)
    #sourceOLS['metaXColumns'] = metaX.columns
    #print sourceOLS
    #print metaX
    return sourceOLS

def initialInstancePredict(df,idx,sourceModels,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    if weightType == 'OLS' or weightType =='OLSFE' or weightType =='OLSFEMI':
        if 'metaXColumns' not in weights:
            historyData = df.loc[:idx].copy()
            historyData = historyData.drop(idx,axis=0)
            if len(historyData)<10:
                weightType = 'Even'
                weights = getEvenWeights(historyData,sourceModels,tLabel,DROP_FIELDS)
            else:
                weights = updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,weightType)

        if not weightType =='Even':
            metaX = pd.DataFrame(columns=weights['metaXColumns'])
    for k,v in sourceModels.items():
        sourceP = np.round(sourceModels[k].predict(X),decimals=1)
        if weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'OLSFEMI':
            print("index is : "+str(idx))
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights[k]/sum(weights.values())
            combo += weight*sourceP
    
    if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'OLSFEMI':
        #print weights['metaModel']
        #print "HEREEEE"
        #print metaX
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
    combo = np.round(combo,decimals=1)
    
    df.loc[idx,'predictions'] = combo
    return df.loc[idx]
'''
def instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    #sourceP = dict()
    for k,v in sourceModels.iteritems():
        sourceP = np.round(sourceModels[k]['model'].predict(X),decimals=1)
        weight = weights['sourceR2s'][k]/weights['totalR2']
        combo += weight*sourceP

    targetP = np.round(targetModel.predict(X),decimals=1)
    combo += (weights['targetR2']/weights['totalR2'])*targetP
    combo = np.round(combo,decimals=1)
    
    df.loc[idx,'predictions'] = combo
    return df.loc[idx]

def initialInstancePredict(df,idx,sourceModels,weights,tLabel,DROP_FIELDS,weightType):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    #sourceP = dict()
    for k,v in sourceModels.iteritems():
        sourceP = np.round(sourceModels[k]['model'].predict(X),decimals=1)
        weight = weights[k]/sum(weights.values())
        combo += weight*sourceP

    combo = np.round(combo,decimals=1)
    
    df.loc[idx,'predictions'] = combo
    return df.loc[idx]
'''
def initialPredict(df,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType):
    for idx in df.index:
        df.loc[idx] = instancePredict(df,idx,sourceModels,targetModel,weights,tLabel,DROP_FIELDS,weightType)
    return df
'''
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    sourceP = np.round(sModel.predict(X),decimals=1)
    targetP = np.round(tModel.predict(X),decimals=1)
    combo = (1-alpha)*targetP + alpha*sourceP
    df['predictions'] = combo

    return df
'''
'''
def updateAlpha(df,sourceModel,targetModel,alpha,tLabel,DROP_FIELDS):
    alphas = []
    if DELTA < alpha and alpha < (1-DELTA):
        alphas = [(alpha-DELTA),alpha,(alpha+DELTA)]
    elif alpha <= DELTA:
        alphas = [alpha,(alpha+DELTA)]
    else:
        #print "3"
        #print alpha
        alphas = [(alpha-DELTA),alpha]
    #print alphas
    errTot,bestAlpha = testAlphas(df,sourceModel,targetModel,alphas,tLabel,DROP_FIELDS)
    #print "update alpha: "+ str(bestAlpha)
    return bestAlpha

def testAlphas(df,sourceModel,targetModel,alphas,tLabel,DROP_FIELDS):
    errTot = 0
    chooseAlpha = alphas[0]
    testDF = df.copy()
    for i in alphas:
        if i < 0 :
            i = 0
        errs = 0
        startIDX = df.index.min()
        endIDX = df.index.max()
        testDF = initialPredict(df,sourceModel,targetModel,i,tLabel,DROP_FIELDS)
        for idx in testDF.index:
            forgettingPow = idx-startIDX
            #print endIDX, startIDX, idx
            #print df
            #print testDF
            #print df.loc[idx,'predictions']
            #print testDF.loc[idx,tLabel]
            
            errs += (FORGETTING_FACTOR**(forgettingPow))*(abs(testDF.loc[idx,'predictions'] - testDF.loc[idx,tLabel])**2)
        if i == alphas[0]:
            errTot = errs
            chooseAlpha = i
        if errTot > errs:
            errTot = errs
            chooseAlpha = i
    #print chooseAlpha
    return errTot, chooseAlpha


def instancePredict(df,idx,sourceModel,targetModel,alpha,tLabel,DROP_FIELDS):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    sourceP = np.round(sourceModel.predict(X),decimals=1)
    targetP = np.round(targetModel.predict(X),decimals=1)
    combo = (1-alpha)*targetP + alpha*sourceP
    df.loc[idx,'predictions'] = combo
    return df.loc[idx]
    
def initialPredict(df,sModel,tModel,alpha,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    sourceP = np.round(sModel.predict(X),decimals=1)
    targetP = np.round(tModel.predict(X),decimals=1)
    combo = (1-alpha)*targetP + alpha*sourceP
    df['predictions'] = combo
    return df
'''
