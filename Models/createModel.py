import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
from sklearn.kernel_approximation import RBFSampler as RBF
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge as KRidge
from sklearn.linear_model import ElasticNet as Elastic
from sklearn.preprocessing import PolynomialFeatures as Polynomial
from sklearn.metrics import r2_score as r2

def createSVR(DEFAULT_PRED):
    if DEFAULT_PRED == 'Following':
        print("using the right model")
        model = SVR(kernel='linear',tol=0.01,epsilon=0.02,C=0.10)#,C=1,class_weight='balanced')
    elif DEFAULT_PRED == 'Heating':
        model = SVR(kernel = 'rbf',epsilon=0.001,gamma=0.001,C=5,tol=0.000001,max_iter=800)
    else:
        model = SVR(kernel='linear',tol = 0.01,epsilon=0.01)#,C=5,max_iter=1500)#,max_iter=800)

    return model

def createElasticNet(DEFAULT_PRED):
    if DEFAULT_PRED == 'Following':
        model = Elastic(alpha=0,l1_ratio=0,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random')
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0,l1_ratio=0,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.01,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.01,l1_ratio=0.8,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=True)),('elasticnet',Elastic(alpha=0.01,l1_ratio=0.8,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=True)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.8,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=True)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=True)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=True)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.1,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.1,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.1,max_iter=200000,positive=False,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=False,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=False,normalize=True,selection='random'))])
        #best so far
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.5,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=False,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.8,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=False,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.7,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=False,normalize=True,selection='random'))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.6,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=False,normalize=True,selection='random'))])
        
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=0.001,l1_ratio=0.7,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=False,normalize=True,selection='random'))])
    elif DEFAULT_PRED == 'Heating':
        # model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0,l1_ratio=0,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        # model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0.001,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        # model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0.001,l1_ratio=0.3,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        # model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0.001,l1_ratio=0.8,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        # model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0.001,l1_ratio=0.5,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        # model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0.001,l1_ratio=0.8,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        # model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0.001,l1_ratio=0.3,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Elastic(alpha=0.001,l1_ratio=0.8,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random'))])
        
    else:
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('elasticnet',Elastic(alpha=1,l1_ratio=0.3,fit_intercept=True,copy_X=True,tol=0.01,max_iter=200000,positive=True,normalize=True,selection='random'))])
        model = Elastic(alpha=0,l1_ratio=0,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random')
        model = Elastic(alpha=0,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random')
        #all 3 work well
        model = Elastic(alpha=0.01,l1_ratio=0.2,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random')
        model = Elastic(alpha=0.01,l1_ratio=0.3,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random')
        model = Elastic(alpha=0.01,l1_ratio=0.5,fit_intercept=True,copy_X=True,tol=0.001,max_iter=200000,positive=True,normalize=True,selection='random')
    return model

def createRidgeReg(DEFAULT_PRED):
    if DEFAULT_PRED == 'Following':
        model = Ridge(alpha= 5,tol=0.01,solver='lsqr')
        model = KRidge(alpha= 10,kernel='linear')#tol=0.01,solver='auto')
        model = KRidge(alpha= 3,kernel='linear')#tol=0.01,solver='auto')
        # model = KRidge(alpha= 1,kernel='linear')#tol=0.01,solver='auto')
    elif DEFAULT_PRED == 'Heating':
        model = Pipeline([('transform',RBF(gamma=0.001)),('ridge',Ridge(alpha= 0.2,tol=0.01,solver='lsqr',normalize=True))])
    else:
        model = Ridge(alpha= 1,tol=0.01,solver='lsqr')
    return model
    

def createPipeline(df,tLabel,DROP_FIELDS,DEFAULT_PRED,LEARNER_TYPE,METASTATS):
    if LEARNER_TYPE == 'MLP':
        create = createMLP
    elif LEARNER_TYPE == 'DTR':
        create = createDT
    elif LEARNER_TYPE == 'Ridge':
        create = createRidgeReg
    elif LEARNER_TYPE == 'Elastic':
        create = createElasticNet
    else:
        create = createSVR
    model = create(DEFAULT_PRED)

    # model = SVR(kernel='rbf',gamma = 0.01,tol = 0.05,C = 0.004,epsilon=0.2)
    X = df.drop(DROP_FIELDS,axis=1).copy()
    y = df[tLabel].copy()
    X = X.drop(tLabel,axis=1)
    # trainX=X[:90]
    # trainy=y[:90]
    # testX=X[91:]
    # testy=y[91:]
    model = model.fit(X,y)
    METASTATS['LOCALMODELSTATS']['modelsLearnt']+=1
    # model = model.fit(trainX,trainy)
    # predicted = model.predict(trainX)
    # trainX['predictions'] = predicted
    # trainX[tLabel]=trainy
    # print(trainX[[tLabel,'predictions']])
    # print(r2(trainX[tLabel],trainX['predictions']))
    # predicted = model.predict(testX)
    # testX['predictions'] = predicted
    # testX[tLabel]=testy
    # print(testX[[tLabel,'predictions']])
    # print(r2(testX[tLabel],testX['predictions']))
    # exit()
    return model,METASTATS


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
    df.loc[idx,'predictions'] = predicted
    return df.loc[idx]

def singleInstancePredict(instance,model,tLabel,DROP_FIELDS):
    X = instance.drop(DROP_FIELDS).copy()
    X = X.drop(tLabel).values.reshape(1,-1)
    Y = instance[tLabel].copy()
    predicted = model.predict(X)[0]
    return predicted

def createDT(DEFAULT_PRED):
    if DEFAULT_PRED == 'Following':
        model=DTR(criterion="mse",max_depth=10)
    elif DEFAULT_PRED == 'Heating':
        model = Pipeline([('transform',RBF(gamma=0.001)),('dtr',DTR(criterion="mse"))])#activation='identity',solver='adam',max_iter=800,alpha=0.02,tol=0.0001,hidden_layer_sizes=(500,)))])
    else: #if DEFAULT_PRED == 'Sudden':
        model=DTR(criterion="mse",max_depth=5,min_samples_split=0.2)
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('dtr',DTR(criterion="mse",splitter="random"))])#,min_impurity_decrease=0.35))])
        # model = Pipeline([('transform',RBF(gamma=2)),('ridge',DTR(criterion='mse',max_depth=10))])
    # else:
        # model=DTR(criterion="mse",max_depth=5)
    return model

def createMLP(DEFAULT_PRED):
    if DEFAULT_PRED == 'Following':
        # model = MLP(activation='identity',solver='adam',max_iter=800,alpha=0.5,tol=0.01,hidden_layer_sizes=(1000,))
        # model = MLP(activation='identity',solver='adam',max_iter=800,alpha=5,tol=0.01,hidden_layer_sizes=(1000,))
        model = MLP(activation='identity',solver='adam',max_iter=800,alpha=0.005,tol=0.00001,hidden_layer_sizes=(100,100))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=0.005,tol=0.00001,hidden_layer_sizes=(50,50))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=0.01,tol=0.00001,hidden_layer_sizes=(50,50))
        #this is best so far
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=0.01,tol=0.00001,hidden_layer_sizes=(50,50,50))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=0.01,tol=0.00001,hidden_layer_sizes=(50,50))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=0.01,tol=0.00001,hidden_layer_sizes=(100,100,100))
        #herereerer
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=0.001,tol=0.00001,hidden_layer_sizes=(100,100,100))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=10,tol=0.00001,hidden_layer_sizes=(100,100,100))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=10,tol=0.001,hidden_layer_sizes=(100,100,100))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=10,tol=0.0001,hidden_layer_sizes=(100,100,100))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=10,tol=0.001,hidden_layer_sizes=(10,10,10,10,10))
        model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=10,tol=0.001,hidden_layer_sizes=(500,500,500))
        #best is relu
        model = MLP(activation='relu',solver='adam',max_iter=2000,alpha=10,tol=0.001,hidden_layer_sizes=(100,100,100))
        model = MLP(activation='relu',solver='adam',max_iter=2000,alpha=10,tol=0.01,hidden_layer_sizes=(100,100,100))
        model = MLP(activation='relu',solver='adam',max_iter=5000,alpha=10,tol=0.001,hidden_layer_sizes=(100,100,100))
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('mlp',MLP(activation='identity',solver='adam',max_iter=800,alpha=0.02,tol=0.0001,hidden_layer_sizes=(500,)))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('mlp',MLP(activation='identity',solver='adam',max_iter=800,alpha=3,tol=0.0001,hidden_layer_sizes=(500,)))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('mlp',MLP(activation='identity',solver='adam',max_iter=800,alpha=10,tol=0.0001,hidden_layer_sizes=(500,)))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('mlp',MLP(activation='identity',solver='adam',max_iter=800,alpha=10,tol=0.0001,hidden_layer_sizes=(1000,)))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('mlp',MLP(activation='relu',solver='adam',max_iter=800,alpha=10,tol=0.0001,hidden_layer_sizes=(1000,)))])
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('mlp',MLP(activation='identity',solver='adam',max_iter=1000,alpha=10,tol=0.0001,hidden_layer_sizes=(1000,)))])
        #best so far
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=False)),('mlp',MLP(activation='identity',solver='adam',max_iter=800,alpha=10,tol=0.0001,hidden_layer_sizes=(500,)))])
        
        model = Pipeline([('transform',Polynomial(degree=1,include_bias=True)),('mlp',MLP(activation='relu',solver='adam',max_iter=20000,alpha=15,tol=0.000001,hidden_layer_sizes=(3000),epsilon=0.001,verbose=True,learning_rate_init=0.001,learning_rate='adaptive'))])
        # model = MLP(activation='identity',solver='adam',max_iter=2000,alpha=0.01,tol=0.00001,hidden_layer_sizes=(100,100,100))
    elif DEFAULT_PRED == 'Heating':
        # model = Pipeline([('transform',RBF(gamma=0.001)),('mlp',MLP(activation='identity',solver='adam',max_iter=800,alpha=0.02,tol=0.00001,hidden_layer_sizes=(800,)))])
        model = Pipeline([('transform',RBF(gamma=0.001)),('mlp',MLP(activation='identity',solver='adam',max_iter=800,alpha=0.02,tol=0.0001,hidden_layer_sizes=(500,)))])
    else:
        # model = MLP(activation='identity',solver='adam',max_iter=800,alpha=0.01,tol=0.001,hidden_layer_sizes=(500,))
        model = MLP(activation='identity',solver='adam',max_iter=800,alpha=0.01,tol=0.001,hidden_layer_sizes=(1000,))
    return model
