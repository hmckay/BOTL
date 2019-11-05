import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
from sklearn.kernel_approximation import RBFSampler as RBF
from sklearn.pipeline import Pipeline

def createPipeline(df,tLabel,DROP_FIELDS):
    # model = SVR(kernel='linear',tol=0.01,epsilon=0.02,C=0.10)#,C=1,class_weight='balanced')
    # model = SVR(kernel='linear',epsilon=0.001,tol=0.0001)
    model = SVR(kernel = 'rbf',epsilon=0.001,gamma=0.001,C=5,tol=0.000001,max_iter=800)
    # model = SVR(kernel='rbf',gamma = 0.01,tol = 0.05,C = 0.004,epsilon=0.2)
    X = df.drop(DROP_FIELDS,axis=1).copy()
    y = df[tLabel].copy()
    X = X.drop(tLabel,axis=1)
    model = model.fit(X,y)
    return model


def initialPredict(df,model,tLabel,DROP_FIELDS):
    X = df.drop(DROP_FIELDS,axis=1).copy()
    X = X.drop(tLabel,axis=1)
    Y = df[tLabel].copy()
    predicted = model.predict(X)
    df['predictions'] = predicted
    return df

def instancePredict(df,idx,model,tLabel,DROP_FIELDS):
    X = df.loc[idx].drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = df.loc[idx,tLabel]#.copy()
    predicted = model.predict(X)[0]
    # print("new predicted: "+str(predicted))
    # predicted = model.predict(X)
    # print("old predicted: "+str(predicted))
    df.loc[idx,'predictions'] = predicted
    # print('RePro instance predict: '+str(df.loc[idx,'predictions']))
    return df.loc[idx]


