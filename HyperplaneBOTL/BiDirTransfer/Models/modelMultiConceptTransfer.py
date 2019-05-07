from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import RidgeCV 
from sklearn.linear_model import Ridge 
from scipy.optimize import nnls
import operator
from sklearn.feature_selection import mutual_info_regression
#DELTA = 0.025
#FORGETTING_FACTOR = 0.995
CULLTHRESH = 0.2
MITHRESH = 0.95

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
    elif weightType =='OLSFEMI':
        return calcOLSFEMIWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS)
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
        #sourceP[k] = np.round(sourceModels[k].predict(X),decimals=3)
        sourceP[k] = sourceModels[k].predict(X)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
        totalR2 += sourceR2[k]
            
    #targetP = np.round(targetModel.predict(X),decimals=3)
    targetP = targetModel.predict(X)
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
    metaX = pd.DataFrame(columns = sourceModels.keys())
    #print X
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    #metaX['target'] = np.round(targetModel.predict(X),decimals=3)
    metaX['target'] = targetModel.predict(X)
    metaModel = OLS()
    metaModel.fit(metaX,Y)
    #print model.coef_
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    #print sourceOLS
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns,'coeffs':metaModel.coef_}
    #print weights
    return weights

def calcNNLSWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    #print X
    
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    
    #metaX['target'] = np.round(targetModel.predict(X),decimals=3)
    metaX['target'] = targetModel.predict(X)
    #metaModel = Ridge(alpha=100)
    #metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)#fit_intercept=False,tol=0.0001,solver='lsqr')
    #metaModel.fit(metaX,Y)
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)#.as_martix())
    #print model.coef_
    
    sourceOLS = dict()
    '''
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    #print sourceOLS
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    '''
    weights = {'sourceR2s':sourceOLS,'targetR2':0, 'totalR2':0, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    #weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    #print weights
    return weights

def calcRidgeWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    #print X
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    #metaX['target'] = np.round(targetModel.predict(X),decimals=3)
    metaX['target'] = targetModel.predict(X)
    #metaModel = Ridge(alpha=100)
    metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)#fit_intercept=False,tol=0.0001,solver='lsqr')
    metaModel.fit(metaX,Y)
    #metaModel,rnorm = nnls(metaX.as_matrix(),Y.as_martix())
    #print model.coef_
    
    sourceOLS = dict()
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    #print sourceOLS
    targetOLS = sourceOLS['target']
    del sourceOLS['target']
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    
    #weights = {'sourceR2s':0,'targetR2':0, 'totalR2':0, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel, 'metaXColumns': metaX.columns}
    #print weights
    return weights

def calcOLSFEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    dropKeys = []
    #print X
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= 0:
            dropKeys.append(k)
    #metaX['target'] = np.round(targetModel.predict(X),decimals=3)
    metaX['target'] = targetModel.predict(X)
    
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
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    #print weights
    return weights

def calcOLSFEMIWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    dropKeys = []
    #print X
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= CULLTHRESH:
            dropKeys.append(k)
    #metaX['target'] = np.round(targetModel.predict(X),decimals=3)
    metaX['target'] = targetModel.predict(X)
    
    for k,v in sourceModels.iteritems():
        if k in dropKeys:
            continue
        for l,u in sourceModels.iteritems():
            if (l in dropKeys) or l==k:
                continue
            predA = sourceModels[k].predict(X)
            predB = sourceModels[l].predict(X)
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
    totalOLS = targetOLS + sum(sourceOLS.itervalues())
    weights = {'sourceR2s':sourceOLS,'targetR2':targetOLS, 'totalR2':totalOLS, 'metaModel': metaModel,'metaXColumns':metaX.columns}
    #print weights
    return weights

def calcMSEWeights(df,sourceModels,targetModel,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceMSE = dict()
    totalMSE = 0
    for k,v in sourceModels.iteritems():
        #sourceP[k] = np.round(sourceModels[k].predict(X),decimals=3)
        sourceP[k] = sourceModels[k].predict(X)
        sourceMSE[k] =  metrics.mean_squared_error(Y,sourceP[k])
        if sourceMSE[k] <= 0:
            sourceMSE[k] = 0.00000000000001
        sourceMSE[k] = 1/sourceMSE[k]
        totalMSE += sourceMSE[k]
            
    #targetP = np.round(targetModel.predict(X),decimals=3)
    targetP = targetModel.predict(X)
    targetMSE = metrics.mean_squared_error(Y,targetP)
    if targetMSE <= 0: 
        targetMSE = 0.00000000000001
    totalMSE += targetMSE
    #ws = (sourceR2/(sourceR2+targetR2))
    #wt = (targetR2/(sourceR2+targetR2))
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
    elif weightType =='OLSFEMI':
        return updateInitialOLSFEMIWeights(df,sourceModels,tLabel,DROP_FIELDS)
    else:
        return updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS)

def updateInitialR2Weights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceR2 = dict()
    #totalR2 = 0
    for k,v in sourceModels.iteritems():
        #sourceP[k] = np.round(sourceModels[k].predict(X),decimals=3)
        sourceP[k] = sourceModels[k].predict(X)
        sourceR2[k] = metrics.r2_score(Y,sourceP[k])
        if sourceR2[k] <= 0:
            sourceR2[k] = 0.00000000000001
        #totalR2 += sourceR2[k]
    
    #targetR2 = 0
    #targetP = np.round(targetModel.predict(X),decimals=3)
    #targetR2 = metrics.r2_score(Y,targetP)
    #if targetR2 <= 0: 
    #    targetR2 = 0.00000000000001
    #totalR2 += targetR2
    #ws = (sourceR2/(sourceR2+targetR2))
    #wt = (targetR2/(sourceR2+targetR2))
    #weights = {'sourceR2s':sourceR2,'targetR2':targetR2, 'totalR2':totalR2}
    return sourceR2

def updateInitialOLSWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    #print X
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
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

def updateInitialRidgeWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    #print X
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    #print metaX
    #metaModel = OLS()
    #metaModel,rnorm = nnls(metaX.as_matrix(),Y.as_martix())
    #metaModel = Ridge(alpha=100)#fit_intercept=False,tol=0.0001,solver='lsqr')
    metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)#fit_intercept=False,tol=0.0001,solver='lsqr')
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

def updateInitialNNLSWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    metaX = pd.DataFrame(columns = sourceModels.keys())
    #print X
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
    #print metaX
    #metaModel = OLS()
    metaModel,rnorm = nnls(metaX.as_matrix(),Y)#.as_martix())
    #metaModel = Ridge(alpha=100)#fit_intercept=False,tol=0.0001,solver='lsqr')
    #metaModel = RidgeCV(alphas = (0.1,1.0,2.0,3.0,5.0,7.0,10.0),fit_intercept=False)#fit_intercept=False,tol=0.0001,solver='lsqr')
    #metaModel.fit(metaX,Y)
    #print model.coef_
    
    sourceOLS = dict()
    '''
    for coef, feat in zip(metaModel.coef_,metaX.columns):
        sourceOLS[feat]=coef
    '''
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
    dropKeys = dict()
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= 0:
            dropKeys[k] = r2
    #print metaX
    if len(dropKeys.keys()) == len(metaX.columns):
        for i in range(0,3):
            if len(dropKeys.keys())>0:
                maxKey = max(dropKeys.iteritems(),key=operator.itemgetter(1))[0]
                del dropKeys[maxKey]
    
    if len(dropKeys.keys()) > 0 and len(dropKeys.keys()) < len(metaX.columns):
        metaX = metaX.drop(dropKeys.keys(),axis=1)
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
    metaX = pd.DataFrame(columns = sourceModels.keys())
    dropKeys = dict()
    for k,v in sourceModels.iteritems():
        #pred = np.round(sourceModels[k].predict(X),decimals=3)
        pred = sourceModels[k].predict(X)
        metaX[k] = pred
        r2 = metrics.r2_score(Y,pred)
        if r2 <= CULLTHRESH:
            dropKeys[k] = r2
    #print metaX
    for k,v in sourceModels.iteritems():
        if k in dropKeys.keys():
            continue
        for l,u in sourceModels.iteritems():
            if (l in dropKeys.keys()) or l==k:
                continue
            predA = sourceModels[k].predict(X)
            predB = sourceModels[l].predict(X)
            combinedpredA = np.stack((predA,np.ones(len(predA)))).T
            mi = mutual_info_regression(combinedpredA,predB)[0]
            if mi > MITHRESH:
                r2A = metrics.r2_score(Y,predA)
                r2B = metrics.r2_score(Y,predB)
                if r2A>r2B:
                    dropKeys[l] = r2B
                else:
                    dropKeys[k] = r2A

    if len(dropKeys.keys()) == len(metaX.columns):
        for i in range(0,3):
            if len(dropKeys.keys())>0:
                maxKey = max(dropKeys.iteritems(),key=operator.itemgetter(1))[0]
                del dropKeys[maxKey]
    
    if len(dropKeys.keys()) > 0 and len(dropKeys.keys()) < len(metaX.columns):
        metaX = metaX.drop(dropKeys.keys(),axis=1)
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

def updateInitialMSEWeights(df,sourceModels,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    
    sourceP = dict()
    sourceMSE = dict()
    #totalR2 = 0
    for k,v in sourceModels.iteritems():
        #sourceP[k] = np.round(sourceModels[k].predict(X),decimals=3)
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
    #print X
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel].copy()
    combo = 0
    if weightType == 'OLS' or weightType =='OLSFE' or weightType == 'OLSFEMI' or weightType == 'Ridge' or weightType == 'NNLS':
        metaX = pd.DataFrame(columns=weights['metaXColumns'])
    #sourceP = dict()
    for k,v in sourceModels.iteritems():
        #sourceP = np.round(sourceModels[k].predict(X),decimals=3)
        sourceP = sourceModels[k].predict(X)
        if weightType == 'OLS' or weightType == 'OLSFE' or weightType == 'OLSFEMI' or weightType == 'Ridge' or weightType =='NNLS':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights['sourceR2s'][k]/weights['totalR2']
            combo += weight*sourceP
            #weight = weights['metaModel']#weights['sourceR2s'][k]
        #else:
        #    weight = weights['sourceR2s'][k]/weights['totalR2']
        #if weightType == 'MSE':
        #    weight = 1 - weight
        #if idx%20 == 0:
        #    print weight
        #combo += weight*sourceP

    #targetP = np.round(targetModel.predict(X),decimals=3)
    targetP = targetModel.predict(X)
    if weightType == 'OLS' or weightType =='OLSFE' or weightType =='OLSFEMI' or weightType == 'Ridge':
        metaX['target'] = targetP
        #print metaX
        #print weights['metaXColumns']
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
        df.loc[idx,'predictions'] = combo
        #print combo
    elif weightType =='NNLS':
        metaX['target'] = targetP
        metaModel = weights['metaModel']
        #print "metamodel: "+str(metaModel)#.item((1,0)))# *  metaX.as_matrix())
        #print metaX.as_matrix().item((0,0))
        #print "metaModel size: "+str(metaModel.size)
        combo = 0
        for w in range(0,metaModel.size):
            if metaModel.size == 1:
                combo += metaModel.item(0) * metaX.as_matrix().item(0,0)
            else:
                #print metaModel
                #print metaModel.item(0)
                #print metaX.as_matrix().item((0,w))
                #print "here"
                combo += metaModel.item(w) * metaX.as_matrix().item((0,w))
        #print combo
        df.loc[idx,'predictions'] = combo
    else:
        combo += (weights['targetR2']/weights['totalR2'])*targetP
        df.loc[idx,'predictions'] = combo
    
    #combo = np.round(combo,decimals=3)
    #df.loc[idx,'predictions'] = combo
    
    #df.loc[idx,'predictions'] = combo
    return df.loc[idx]

def getEvenWeights(df,sourceModels,tLabel,DROP_FIELDS):
    metaX = pd.DataFrame(columns = sourceModels.keys())
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
    if weightType == 'OLS' or weightType == 'Ridge' or weightType =='OLSFE' or weightType == 'OLSFEMI' or weightType == 'NNLS':
        if 'metaXColumns' not in weights:
            historyData = df.loc[:idx].copy()
            #print historyData
            historyData = historyData.drop(idx,axis=0)
            #print len(historyData)
            if len(historyData)<10:
                weightType = 'Even'
                weights = getEvenWeights(historyData,sourceModels,tLabel,DROP_FIELDS)
                #weights['metaXColumns'] = ['target']
            else:
                weights = updateInitialWeights(historyData,sourceModels,tLabel,DROP_FIELDS,weightType,CULLTHRESH,MITHRESH)

        #print weights
        if not weightType =='Even':
            metaX = pd.DataFrame(columns=weights['metaXColumns'])

    #sourceP = dict()
    #print "NEW RUNNNNNN"
    #print "about to print sourcemodels"
    #print sourceModels
    for k,v in sourceModels.iteritems():
        print "k is: "+str(k)
        #sourceP = np.round(sourceModels[k].predict(X),decimals=3)
        sourceP = sourceModels[k].predict(X)
        if weightType == 'OLS' or weightType == 'Ridge' or weightType == 'OLSFE' or weightType=='OLSFEMI' or weightType == 'NNLS':
            if k in metaX.columns:
                metaX[k] = sourceP
        else:
            weight = weights[k]/sum(weights.values())
            combo += weight*sourceP
            #weight = weights[k]
        #else:
        #    weight = weights[k]/sum(weights.values())
        #if weightType == 'MSE':
        #    weight = 1 - weight
        #combo += weight*sourceP

    
    if weightType == 'OLS' or weightType =='OLSFE' or weightType=='OLSFEMI' or weightType == 'Ridge':
        #metaX['target'] = targetP
        #print metaX
        #print weights['metaXColumns']
        metaModel = weights['metaModel']
        combo = metaModel.predict(metaX)
    elif weightType =='NNLS':
        #metaX['target'] = targetP
        metaModel = weights['metaModel']
        combo = 0
        for w in range(0,metaModel.size):
            if metaModel.size == 1:
                combo += metaModel.item(0) * metaX.as_matrix().item(0,0)
            else:
                combo += metaModel.item(w) * metaX.as_matrix().item((0,w))
        #df.loc[idx,'predictions'] = combo
        #print combo

        #print combo
    #combo = np.round(combo,decimals=3)
    
    df.loc[idx,'predictions'] = combo
    return df.loc[idx]

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
