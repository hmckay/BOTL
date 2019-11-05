import numpy as np
from numpy.linalg import svd as SVD
import math
from scipy.spatial.distance import euclidean as euclid
from Models.stsc_ulti import affinity_to_lap_to_eig, reformat_result, get_min_max
from Models.stsc_np import get_rotation_matrix as get_rotation_matrix_np
from Models.stsc_autograd import get_rotation_matrix as get_rotation_matrix_autograd
from Models.stsc_manopt import get_rotation_matrix as get_rotation_matrix_manopt

def principalAngles(x,y):
    xT = x.transpose()
    xTy = np.dot(xT,y)
    u,sig,v = SVD(xTy)
    angles = np.zeros(len(sig))
    print("enumerating sig:")
    for idx,a in enumerate(sig):
        print(a)
        if a>=1:
            angles[idx] = 0
        else:
            angles[idx] = np.arccos(a)
    print("x and y are:")
    print(x.shape)
    print(y.shape)
    print("sig")
    print(sig)
    print("angles:")
    print((np.sum(np.cos(angles))))
    print((1/len(angles)))
    print(1 - (1/len(angles))*(np.sum(np.cos(angles))))#, (1 - (1-len(angles))*(np.sum(np.cos(angles)**2)))
    print(np.cos(angles))
    print(len(sig))
    return (1 - (1/len(angles))*(np.sum(np.cos(angles))))#, (1 - (1-len(angles))*(np.sum(np.cos(angles)**2)))

# def principalAnglesNorm(x,y):
    # xT = x.transpose()
    # xTy = np.dot(xT,y)
    # u,sig,v = SVD(xTy)
    # aff = sum(sig**2)/len(sig)
    # aff = math.sqrt(aff)
    # return aff

#affinity metric = euclid or ...
def calcAffinity(base,k,affinityMetric):
    affinityMatrix = np.zeros((len(base),len(base)))
    # affinityMatrix2 = np.zeros((len(base),len(base)))
    distanceMatrix = np.zeros((len(base),len(base)))
    # distanceMatrix2 = np.zeros((len(base),len(base)))
    kNN= [k if len(base)>k else len(base)-1]

    for idx,i in enumerate(base):
        distances = np.zeros(len(base))
        # distances2 = np.zeros(len(base))
        for jdx,j in enumerate(base):
            # distances[jdx],distances2[jdx] = affinityMetric(i,j)
            distances[jdx] = affinityMetric(i,j)
            print("i is:")
            print(i)
            print("j is:")
            print(j)
            # print(str(idx)+","+str(jdx)+": "+str(distances[jdx]))
        distanceMatrix[idx]=distances
        # distanceMatrix2[idx]=distances2

    # if affinityMetric == principalAnglesNorm:
        # print("returning distance matrix")
    # return distanceMatrix

    for idx, i in enumerate(distanceMatrix):
        affinities = np.zeros(len(base))
        # affinities2 = np.zeros(len(base))
        sortedI = i.copy()
        # sortedI2 = distanceMatrix2[idx].copy()
        np.ndarray.sort(sortedI)
        # np.ndarray.sort(sortedI2)
        iNormaliser = sortedI[kNN]
        # iNormaliser2 = sortedI2[kNN]
        for jdx, j in enumerate(i):
            sortedJ = distanceMatrix[jdx].copy()
            # sortedJ2 = distanceMatrix2[jdx].copy()
            np.ndarray.sort(sortedJ)
            # np.ndarray.sort(sortedJ2)
            jNormaliser = sortedJ[kNN]
            # jNormaliser2 = sortedJ2[kNN]
            affinities[jdx] = math.exp(-(i[jdx]**2)/(iNormaliser*jNormaliser))
            # affinities2[jdx] = math.exp(-(distanceMatrix2[idx][jdx]**2)/(iNormaliser2*jNormaliser2))
        affinityMatrix[idx] = affinities
        # affinityMatrix2[idx] = affinities2

    # if len(affinityMatrix) > 5:
    print("originaldistance")
    print(distanceMatrix)
    # print("squareddistance")
    # print(distanceMatrix2)
    print("originalAffinity")
    print(affinityMatrix)
    # print("squaredAffinity")
    # print(affinityMatrix2)
    return affinityMatrix#,affinityMatrix2

def STSC(metaX,affinityMetric,targetModelName,k):
    if affinityMetric == 'euclid':
        metric = euclid
        names = metaX.columns.tolist()
        predMatrixT = metaX.to_numpy(copy=True)
        base = predMatrixT.transpose()
    else:
        metric = principalAngles
        names = metaX.keys()
        base = []
        print("names:")
        print(names)
        for i in names:
            base.append(metaX[i])

    # similarity_matrix,similarity_matrix2 = calcAffinity(base,k,metric)
    similarity_matrix = calcAffinity(base,k,metric)
    print(names)    
    # if len(names())> 5:
        # minClusters = 2
    # else:
    minClusters = None
    # groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix,similarity_matrix2, names, get_rotation_matrix_np, minClusters, None)
    groupedName, groupedID = self_tuning_spectral_clustering(similarity_matrix, names, get_rotation_matrix_np, minClusters, None)
    print(groupedName)
    # print(groupedID)

    # clusteredNames = groupedName[np.where(groupedName==targetModelName)]
    clusteredNames = [x for x in groupedName if targetModelName in x][0]

    return groupedName


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
    re = []
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r = get_rotation_matrix(x, c)
        re.append((cost, x.dot(r)))
        print('n_cluster: %d \t cost: %f' % (c, cost))
    COST, Z = sorted(re, key=lambda x: x[0])[0]
    if len(affinity) > 10:
        print("original clustering")
        print(reformat_result(np.argmax(Z, axis=1), Z.shape[0],names))
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
