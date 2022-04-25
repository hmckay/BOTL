import numpy as np
from numpy.linalg import svd as SVD
import math
import pandas as pd
from scipy.spatial.distance import euclidean as euclid
from Models.stsc_ulti import affinity_to_lap_to_eig, reformat_result, get_min_max
from Models.stsc_np import get_rotation_matrix as get_rotation_matrix_np
from Models.stsc_autograd import get_rotation_matrix as get_rotation_matrix_autograd
from Models.stsc_manopt import get_rotation_matrix as get_rotation_matrix_manopt

def isOrthonormal(x):
    vectors = x.shape[1]
    # print("testing orthonormal")

    for i in range(0,vectors):
        a = x[:,i]
        for j in range(0,vectors):
            # print(str(i)+","+str(j))
            b = x[:,j]
            # print(np.dot(a,b))
            # print(np.round(np.dot(a,b),decimals=14))

def isNormal(x):
    vectors = x.shape[1]
    # print("testing vector length")

    for i in range(0,vectors):
        a = x[:,i]
        mag = math.sqrt(sum(idx**2 for idx in a))
        # print("colum vec:"+str(a))
        # print(mag)

def normalise(x):
    normDF = pd.DataFrame(x)
    for col in normDF.columns:
        normDF[col] = normDF[col]-normDF[col].mean()
        std = normDF[col].std()
        if std != pd.np.nan and std != 0:
            normDF[col] = normDF[col]/normDF[col].std()
            # print("normalised "+str(col))
        # else:
            # print("COULDN'T NORMALISE")
    x = normDF.to_numpy()
    return x


def principalAngles(x,y):
    swapped = False
    if y.shape[1] < x.shape[1]:
        swapped=True
        # temp = x
        # x = y
        # y = temp
    
    # isOrthonormal(x)
    # isOrthonormal(y)
    # isNormal(x)
    # isNormal(y)
    yshape = y.shape[1]
    # exit()
    # x = normalise(x)
    # y = normalise(y)
    if y.shape[1] < x.shape[1]:
        yT = y.transpose()
        yTx = np.dot(yT,x)
        u,sig,v = SVD(yTx)
        angles = np.zeros(len(sig))
        yshape = x.shape[1]
    else:
        xT = x.transpose()
        xTy = np.dot(xT,y)
        u,sig,v = SVD(xTy)
        angles = np.zeros(len(sig))
    # angles = np.zeros(len(sig),dtype=np.longdouble)
    # print("enumerating sig:")
    # print(x)
    # print(y)
    # print("xTy:"+str(xTy))
    # print("sig:"+str(sig))
    # print("shape of x: "+str(x.shape[1]))
    # print("shape of y: "+str(y.shape[1]))
    # print("swapped: "+str(swapped))
    for idx,a in enumerate(sig):
        ang = a#np.round(a,decimals=14)
        # print(idx,a)
        if a>=1:
            angles[idx] = np.arccos(1)
        else:
            angles[idx] = np.arccos(ang)
    # print("x and y are:")
    # print(x.shape)
    # print(y.shape)
    # print("sig")
    # print(sig)
    # print("angles:")
    # print(len(angles))
    # print(angles)
    tot = 0
    for a in angles:
        # print(np.cos(a))#/len(angles))
        # tot += np.cos(a)/len(angles)
        tot += np.cos(a)/yshape
    # print(tot.dtype)
    # print(tot)
    # print(np.cos(angles[-1]))
    # print((np.sum(np.cos(angles))))
    # print((1/len(angles)))
    # print("should be:")
    # print(1-tot)
    # print("actually is being used:")
    # print(1 - (np.sum(np.cos(angles)/len(angles))))#, (1 - (1-len(angles))*(np.sum(np.cos(angles)**2)))

    # print(1 - (1/len(angles))*(np.sum(np.cos(angles))))#, (1 - (1-len(angles))*(np.sum(np.cos(angles)**2)))
    # print(np.cos(angles))
    # print(len(sig))
    return (1 - tot)#(np.sum(np.cos(angles)/len(angles))))#, (1 - (1-len(angles))*(np.sum(np.cos(angles)**2)))
    # return (1 - (1/len(angles))*(np.sum(np.cos(angles))))#, (1 - (1-len(angles))*(np.sum(np.cos(angles)**2)))

# def principalAnglesNorm(x,y):
    # xT = x.transpose()
    # xTy = np.dot(xT,y)
    # u,sig,v = SVD(xTy)
    # aff = sum(sig**2)/len(sig)
    # aff = math.sqrt(aff)
    # return aff

#affinity metric = euclid or ...
def pruneAffinity(distanceMatrix,affinityMatrix,base,cullingThreshold,currentModID):
    toRemove = []
    for i in affinityMatrix.keys():
        for j in affinityMatrix.keys():
            if affinityMatrix[i][j]>cullingThreshold and j>i and j != currentModID:
                base[j]['pruned']=1
                toRemove.append(j)

    for r in toRemove:
        del distanceMatrix[r]
        del affinityMatrix[r]
        for u in affinityMatrix.keys():
            del distanceMatrix[u][r]
            del affinityMatrix[u][r]


    return distanceMatrix,affinityMatrix,base

# def calcPrunedAffinity(base,k,affinityMetric,cullingThreshold,distanceMatrix=None,affinityMatrix=None,names=None,newTarget=True,targetModelName=0):
def calcPrunedAffinity(base,k,affinityMetric,cullingThreshold,distanceMatrix=None,affinityMatrix=None,names=None,newTarget=True,currentModID=0):
    if distanceMatrix is None: distanceMatrix = dict()
    if affinityMatrix is None: affinityMatrix = dict()
    
    unstable = [m for m in list(distanceMatrix.keys()) if m not in list(base.keys())]
    toPrune = [m for m in list(base.keys()) if (base[m]['prune']==True and m in list(distanceMatrix.keys()))]
    toRemove = list(set(unstable+toPrune))
    for u in toRemove:
        del distanceMatrix[u]
        del affinityMatrix[u]
        for s in distanceMatrix.keys():
            del distanceMatrix[s][u]
        for s in affinityMatrix.keys():
            del affinityMatrix[s][u]
        if u in names:
            name.remove(u)
    
    for i in names:
        if i == currentModID and newTarget:
            distanceMatrix[i]=dict()
            for j in distanceMatrix.keys():
                # print("===================")
                # print("getting: "+str(i)+","+str(j))
                distance = affinityMetric(base[i],base[j])
                distanceMatrix[i][j] = distance
                distanceMatrix[j][i] = distance
            distanceMatrix[i][i] = affinityMetric(base[i],base[i])

        if i not in distanceMatrix.keys():
            distanceMatrix[i]=dict()
            for j in distanceMatrix.keys():
                # print("(i,j): ("+str(i)+","+str(j)+")")
                distance = affinityMetric(base[i],base[j])
                distanceMatrix[i][j] = distance
                distanceMatrix[j][i] = distance
            distanceMatrix[i][i] = affinityMetric(base[i],base[i])
    for n in names:
        if (n not in affinityMatrix.keys()) or newTarget:
            for i in names:
                affinityMatrix[i] = dict()
                iZeroNeighbours = sum(1 for vals in distanceMatrix[i].values() if vals==0)
                # print("iZero"+str(iZeroNeighbours))
                # print("base"+str(len(base)))
                if k > iZeroNeighbours and k < len(base):
                    kNN = k
                elif k <= iZeroNeighbours and iZeroNeighbours < len(base):
                    kNN = iZeroNeighbours
                else:
                    kNN = len(base)-1

                # print("distanceMatrix before isort:"+str(distanceMatrix[i]))
                # print(names)
                # print(i)
                # print(kNN)
                iNormaliser = sorted(distanceMatrix[i].values(),reverse=False)[kNN]
                # print("distanceMatrix after isort:"+str(distanceMatrix[i]))
                # print("inorm for "+str(i)+": "+str(iNormaliser))
                for j in names:
                    jZeroNeighbours = sum(1 for vals in distanceMatrix[j].values() if vals==0)
                    if k > jZeroNeighbours and k < len(base):
                        kNN = k
                    elif k <= jZeroNeighbours and jZeroNeighbours < len(base):
                        kNN = jZeroNeighbours
                    else:
                        kNN = len(base)-1
                    
                    jNormaliser = sorted(distanceMatrix[i].values(),reverse=False)[kNN]
                    if iNormaliser == 0 and jNormaliser == 0:
                        iNormaliser = 1
                        jNormaliser = 1
                    elif iNormaliser == 0:
                        iNormaliser = jNormaliser
                    elif jNormaliser == 0:
                        jNormaliser = iNormaliser

                    affinityMatrix[i][j] =  math.exp(-(distanceMatrix[i][j]**2)/(iNormaliser*jNormaliser))
                    
    # print("distance matrix")
    # print(distanceMatrix)
    # print("affinity matrix")
    # print(affinityMatrix)

    distanceMatrix,affinityMatrix,base = pruneAffinity(distanceMatrix,affinityMatrix,base,cullingThreshold,currentModID)


    return distanceMatrix,affinityMatrix,base

def calcAffinity(base,k,affinityMetric,METASTATS,distanceMatrix=None,affinityMatrix=None,names=None,newTarget=True,targetModelName=0):
    # affinityMatrix2 = np.zeros((len(base),len(base)))
    if distanceMatrix is None: distanceMatrix = dict()
    if affinityMatrix is None: affinityMatrix = dict()
    totalCalcs = 0
    tc=0
    discarded = [m for m in list(affinityMatrix.keys()) if m not in list(distanceMatrix.keys())]
    for u in discarded:
        # del distanceMatrix[u]
        del affinityMatrix[u]
        # for s in distanceMatrix.keys():
            # del distanceMatrix[s][u]
        for s in affinityMatrix.keys():
            print("trying to delete: "+str(s)+","+str(u))
            del affinityMatrix[s][u]
    
    unstable = [m for m in list(distanceMatrix.keys()) if m not in list(base.keys())]
    print("unstable:"+str(unstable))
    print("distanceMatrix keys:"+str(distanceMatrix.keys()))
    print("affinityMatrix keys:"+str(affinityMatrix.keys()))
    for u in unstable:
        del distanceMatrix[u]
        del affinityMatrix[u]
        for s in distanceMatrix.keys():
            del distanceMatrix[s][u]
        for s in affinityMatrix.keys():
            print("trying to delete: "+str(s)+","+str(u))
            del affinityMatrix[s][u]

    # print("distanceMatrix: "+str(distanceMatrix))
    # print("affinityMatrix: "+str(affinityMatrix))
    # print("bases: "+str(base.keys()))
    # print("names: "+str(names))
    # print(names)
    # print(base.keys())
    for i in names:
        if i == targetModelName and newTarget:
            distanceMatrix[i]=dict()
            for j in distanceMatrix.keys():
                # print("===================")
                # print("getting: "+str(i)+","+str(j))
                distance = affinityMetric(base[i],base[j])
                distanceMatrix[i][j] = distance
                distanceMatrix[j][i] = distance
                METASTATS['COMPSTATS']['PADistCalc']+=1
                tc+=1
            distanceMatrix[i][i] = affinityMetric(base[i],base[i])

        if i not in distanceMatrix.keys():
            distanceMatrix[i]=dict()
            for j in distanceMatrix.keys():
                # print("(i,j): ("+str(i)+","+str(j)+")")
                distance = affinityMetric(base[i],base[j])
                distanceMatrix[i][j] = distance
                distanceMatrix[j][i] = distance
                METASTATS['COMPSTATS']['PADistCalc']+=1
                tc+=1
            distanceMatrix[i][i] = affinityMetric(base[i],base[i])
    for n in names:
        if (n not in affinityMatrix.keys()) or newTarget:
            for i in names:
                affinityMatrix[i] = dict()
                iZeroNeighbours = sum(1 for vals in distanceMatrix[i].values() if vals==0)
                # iNonZeroNeigh = dict(filter(lambda elem: elem[1]!=0,distanceMatrix[i].items()))
                # print("iZero"+str(iZeroNeighbours))
                # print("base"+str(len(base)))
                if k > iZeroNeighbours and k < len(base):
                    kNN = k
                elif k <= iZeroNeighbours and iZeroNeighbours < len(base):
                    kNN = iZeroNeighbours
                else:
                    kNN = len(base)-1

                # kNN= k if len(base)>k else len(base)-1
                # kNN= k if len(iNonZeroNeigh)>k else len(iNonZeroNeigh)-1
                # print("distance matrix row is: "+str(sorted(distanceMatrix[i].values(),reverse=True)))
                # print("knn"+str(kNN))
                # print("distanceMatrix before isort:"+str(distanceMatrix[i]))
                # print(names)
                # print(i)
                # print(kNN)
                iNormaliser = sorted(distanceMatrix[i].values(),reverse=False)[kNN]
                # if kNN < 0:
                    # iNormaliser = 0
                # else:
                    # iNormaliser = sorted(iNonZeroNeigh.values(),reverse=False)[kNN]
                # print("distanceMatrix after isort:"+str(distanceMatrix[i]))
                # print("inorm for "+str(i)+": "+str(iNormaliser))
                # if iNormaliser == 0:
                    # iNormaliser = 1
                for j in names:
                    jZeroNeighbours = sum(1 for vals in distanceMatrix[j].values() if vals==0)
                    # iNonZeroNeigh = dict(filter(lambda elem: elem[1]!=0,distanceMatrix[i].items()))
                    if k > jZeroNeighbours and k < len(base):
                        kNN = k
                    elif k <= jZeroNeighbours and jZeroNeighbours < len(base):
                        kNN = jZeroNeighbours
                    else:
                        kNN = len(base)-1
                    
                    jNormaliser = sorted(distanceMatrix[i].values(),reverse=False)[kNN]

                    # jNonZeroNeigh = dict(filter(lambda elem: elem[1]!=0,distanceMatrix[j].items()))
                    # kNN= k if len(jNonZeroNeigh)>k else len(jNonZeroNeigh)-1
                    # if kNN < 0:
                        # jNormaliser = 0
                    # else:
                        # jNormaliser = sorted(jNonZeroNeigh.values(),reverse=False)[kNN]
                    
                    # print("distanceMatrix before jsort:"+str(distanceMatrix[j]))
                    # jNormaliser = sorted(distanceMatrix[j].values(),reverse=False)[kNN]
                    # print("distanceMatrix after jsort:"+str(distanceMatrix[j]))
                    # print("jnorm for "+str(j)+": "+str(jNormaliser))
                    # if jNormaliser == 0:
                        # jNormaliser = 1
                    # if iNormaliser == 0:
                        # iNormaliser = jNormaliser
                    # elif jNormaliser == 0:
                        # jNormaliser = iNormaliser
                    # if iNormaliser == 0 or jNormaliser == 0:
                        # affinityMatrix[i][j] =  1#math.exp(-(distanceMatrix[i][j]**2)/(iNormaliser*jNormaliser))
                    # else:
                        # affinityMatrix[i][j] =  math.exp(-(distanceMatrix[i][j]**2)/(iNormaliser*jNormaliser))
                    if iNormaliser == 0 and jNormaliser == 0:
                        iNormaliser = 1
                        jNormaliser = 1
                    elif iNormaliser == 0:
                        iNormaliser = jNormaliser
                    elif jNormaliser == 0:
                        jNormaliser = iNormaliser

                    affinityMatrix[i][j] =  math.exp(-(distanceMatrix[i][j]**2)/(iNormaliser*jNormaliser))
                    METASTATS['COMPSTATS']['PAAffCalc']+=1
                    # affinityMatrix[i][j] =  math.exp(-(distanceMatrix[i][j]**2))#/(iNormaliser*jNormaliser))
                    
    # print("distance matrix")
    # print(distanceMatrix)
    # print("affinity matrix")
    # print(affinityMatrix)
    return distanceMatrix,affinityMatrix,tc,METASTATS
    
    # similarityMatrix = np.zeros((len(base),len(base)))


            # affinities = np.zeros(len(base))
            # # affinities2 = np.zeros(len(base))
            # sortedI = i.copy()
            # # sortedI2 = distanceMatrix2[idx].copy()
            # np.ndarray.sort(sortedI)
            # # np.ndarray.sort(sortedI2)
            # iNormaliser = sortedI[kNN]
            # # iNormaliser2 = sortedI2[kNN]
            # for jdx, j in enumerate(i):
                # sortedJ = distanceMatrix[jdx].copy()
                # # sortedJ2 = distanceMatrix2[jdx].copy()
                # np.ndarray.sort(sortedJ)
                # # np.ndarray.sort(sortedJ2)
                # jNormaliser = sortedJ[kNN]
                # # jNormaliser2 = sortedJ2[kNN]
                # print("tryingtodo: "+str((idx,jdx)))
                # print("affinities:" +str(affinities))
                # print("i"+str(i))
                # affinities[jdx] = math.exp(-(i[jdx]**2)/(iNormaliser*jNormaliser))
                # # affinities2[jdx] = math.exp(-(distanceMatrix2[idx][jdx]**2)/(iNormaliser2*jNormaliser2))
                # print("diddooo: "+str((idx,jdx)))
            # affinityMatrix[idx] = affinities

    # if distanceMatrix is not None and newBases:
        # oldDistMatrix = distanceMatrix.copy()
        # distanceMatrix = dict()#np.zeros((len(base),len(base)))
        # print("old distance to new: "+str((oldDistMatrix.shape,distanceMatrix.shape)))
        # for idx,i in enumerate(oldDistMatrix):
            # #distances = np.pad(i,(0,len(newBases)),'constant',constant_values=(0,))
            # for jdx,j in enumerate(newBases):
                # distances[len(oldDistMatrix)+jdx] = affinityMetric(base[idx],j)
            # distanceMatrix[idx] = distances#[len(i)+jdx] = affinityMetric(i,j)
        # for idx,i in enumerate(newBases):
            # distances = np.zeros(len(base))
            # for jdx,j in enumerate(base):
                # distances[jdx] = affinityMetric(i,j)
            # distanceMatrix[len(oldDistMatrix)+idx]=distances
    # elif distanceMatrix is None:
        # distanceMatrix = np.zeros((len(base),len(base)))
        # # distanceMatrix2 = np.zeros((len(base),len(base)))

        # for idx,i in enumerate(base):
            # distances = np.zeros(len(base))
            # # distances2 = np.zeros(len(base))
            # for jdx,j in enumerate(base):
                # # distances[jdx],distances2[jdx] = affinityMetric(i,j)
                # distances[jdx] = affinityMetric(i,j)
                # # print("i is:")
                # # print(i)
                # # print("j is:")
                # # print(j)
                # # print(str(idx)+","+str(jdx)+": "+str(distances[jdx]))
            # distanceMatrix[idx]=distances
            # # distanceMatrix2[idx]=distances2

    # # if affinityMetric == principalAnglesNorm:
        # # print("returning distance matrix")
    # # return distanceMatrix
    
    # print ("before calc afinities: "+str(distanceMatrix))
    # print ("before calc afinities: "+str(affinityMatrix))
    # print ("before calc base afinities: "+str(len(base)))
    # if newTarget or affinityMatrix is None:
        # affinityMatrix = np.zeros((len(base),len(base)))
        # kNN= [k if len(base)>k else len(base)-1]
        # for idx, i in enumerate(distanceMatrix):
            # affinities = np.zeros(len(base))
            # # affinities2 = np.zeros(len(base))
            # sortedI = i.copy()
            # # sortedI2 = distanceMatrix2[idx].copy()
            # np.ndarray.sort(sortedI)
            # # np.ndarray.sort(sortedI2)
            # iNormaliser = sortedI[kNN]
            # # iNormaliser2 = sortedI2[kNN]
            # for jdx, j in enumerate(i):
                # sortedJ = distanceMatrix[jdx].copy()
                # # sortedJ2 = distanceMatrix2[jdx].copy()
                # np.ndarray.sort(sortedJ)
                # # np.ndarray.sort(sortedJ2)
                # jNormaliser = sortedJ[kNN]
                # # jNormaliser2 = sortedJ2[kNN]
                # print("tryingtodo: "+str((idx,jdx)))
                # print("affinities:" +str(affinities))
                # print("i"+str(i))
                # affinities[jdx] = math.exp(-(i[jdx]**2)/(iNormaliser*jNormaliser))
                # # affinities2[jdx] = math.exp(-(distanceMatrix2[idx][jdx]**2)/(iNormaliser2*jNormaliser2))
                # print("diddooo: "+str((idx,jdx)))
            # affinityMatrix[idx] = affinities
            # # affinityMatrix2[idx] = affinities2

    # # if len(affinityMatrix) > 5:
    # # print("originaldistance")
    # # print(distanceMatrix)
    # # print("squareddistance")
    # # print(distanceMatrix2)
    # # print("originalAffinity")
    # # print(affinityMatrix)
    # # print("squaredAffinity")
    # # print(affinityMatrix2)
    # return similarityMatrix,distanceMatrix,affinityMatrix

def getSimMatrix(affDict,names,targetModelName):
    simKeys = list(names)
    # if targetModelName in simKeys:
        # simKeys.remove(targetModelName)
    simMatrix = np.zeros((len(simKeys),len(simKeys)))
    # print("simkeys:"+str(simKeys))

    for idx,i in enumerate(simKeys):
        similarity = np.zeros(len(simKeys))
        for jdx,j in enumerate(simKeys):
            similarity[jdx]=affDict[i][j]
            # print("("+str(idx)+","+str(jdx)+")"+str(similarity[jdx]))
        simMatrix[idx]=similarity

    return simMatrix
    


def STSC(metaX,affinityMetric,targetModelName,k,METASTATS,distanceMatrix=None,affinityMatrix=None,newTarget=True):
    if affinityMetric == 'euclid':
        metric = euclid
        names = metaX.keys()
        # names = metaX.columns.tolist()
        # predMatrixT = metaX.to_numpy(copy=True)
        # base = predMatrixT.transpose()
    else:
        metric = principalAngles
        names = metaX.keys()
        # print("names:"+str(names))
        # simKeys = list(names)[:]
        # if targetModelName in names:
            # # print("targetModel is:"+str(targetModelName))
            # simKeys.remove(targetModelName)
        # base = dict()#[]
        # newBases = []
        # if existingBases is None:
            # existingBases = []
        # print("Targetnames:"+str(targetModelName))
        # print(names)
        # for i in names:
            # base[i] = metaX[i]
            # # if i != 'target' and (i not in existingBases):
                # # newBases.append(metaX[i])
                # # existingBases.append(i)
            # # if i == 'target' and (targetModelName not in existingBases):
                # # newBases.append(metaX[i])
                # # existingBases.append(i)

            # # if i != targetModelName:
                # # if (i not in existingBases):
                    # # newBases.append(metaX[i])
                    # # existingBases.append(i)
                # base.append(metaX[i])

    # similarity_matrix,similarity_matrix2 = calcAffinity(base,k,metric)
    distanceMatrix,affinityMatrix,tc,METASTATS = calcAffinity(metaX,k,metric,METASTATS,distanceMatrix,affinityMatrix,
            names,newTarget,targetModelName)
    # print("distanceMatrix is:"+str(distanceMatrix))
    # print("names are:"+str(names)+str(targetModelName))
    
    similarity_matrix = getSimMatrix(affinityMatrix,names,targetModelName)
    # print("sim matrix:"+str(similarity_matrix))
    
    # print("newBases"+str(newBases))
    # print("existingBases:"+str(existingBases))
    # print("distanceMatrix is:"+str(distanceMatrix))
    # print(names)    
    # if len(names())> 5:
        # minClusters = 2
    # else:
    minClusters = None
    # groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix,similarity_matrix2, names, get_rotation_matrix_np, minClusters, None)
    # print("simMatrix"+str(targetModelName))
    # print(similarity_matrix)
    groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix, names, get_rotation_matrix_np, minClusters, None)
    print(groupedName)
    # print(groupedID)

    # clusteredNames = groupedName[np.where(groupedName==targetModelName)]
    # clusteredNames = [x for x in groupedName if targetModelName in x][0]

    return groupedName,distanceMatrix,affinityMatrix,tc,METASTATS


# def self_tuning_spectral_clustering(affinity, affinity2,names, get_rotation_matrix, min_n_cluster=None, max_n_cluster=None):
def self_tuning_spectral_clustering(affinity, names, get_rotation_matrix, min_n_cluster=None, max_n_cluster=None):
    # if len(affinity) > 5:
        # w, v = affinity_to_lap_to_eig(affinity2)
        # min_n_cluster, max_n_cluster = get_min_max(w, min_n_cluster, max_n_cluster)
        # re = []
        # for c in range(min_n_cluster, max_n_cluster + 1):
            # x = v[:, -c:]
            # cost, r = get_rotation_matrix(x, c)
            # re.append((cost, x.dot(r)))
            # print('n_cluster: %d \t cost: %f' % (c, cost))
        # COST, Z = sorted(re, key=lambda x: x[0])[0]
        # print("squared clustering")
        # print(reformat_result(np.argmax(Z, axis=1), Z.shape[0],names))
    
    w, v = affinity_to_lap_to_eig(affinity)
    min_n_cluster, max_n_cluster = get_min_max(w, min_n_cluster, max_n_cluster)
    if max_n_cluster > 10:
        max_n_cluster = 10
    re = []
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r = get_rotation_matrix(x, c)
        re.append((cost, x.dot(r)))
        print('n_cluster: %d \t cost: %f' % (c, cost))
    COST, Z = sorted(re, key=lambda x: x[0])[0]
    # if len(affinity) > 10:
        # print("original clustering")
        # print(reformat_result(np.argmax(Z, axis=1), Z.shape[0],names))
    # groupedID,groupedName = reformat_result(np.argmax(Z, axis=1), Z.shape[0],names)
    # print(groupedName)
    # return groupedID
    return reformat_result(np.argmax(Z, axis=1), Z.shape[0],names)


def self_tuning_spectral_clustering_np(affinity, names, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, names, get_rotation_matrix_np, min_n_cluster, max_n_cluster)


def self_tuning_spectral_clustering_autograd(affinity, names, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, names, get_rotation_matrix_autograd, min_n_cluster, max_n_cluster)


def self_tuning_spectral_clustering_manopt(affinity, names, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, names, get_rotation_matrix_manopt, min_n_cluster, max_n_cluster)
