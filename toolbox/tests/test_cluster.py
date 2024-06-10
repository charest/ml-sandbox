import pytest

import toolbox.cluster as cluster
import toolbox.file_utils as futils
import toolbox.math_utils as mutils

import random
import numpy as np


TEST_DATA_DIR = futils.dirname(__file__)

testdata = [
    (False, 2, [118, 132], [0.3305, 0.3333], 83),
    (True,  2, [224,  26], [0.2902, 0.6923], 83),
]


class Patient(cluster.Example):
    pass

###############################################################################

def getData(toScale = False):
    """read in data"""
    hrList, stElevList, ageList, prevACSList, classList = [],[],[],[],[]
    cardiacData = open(TEST_DATA_DIR/'cardiacData.txt', 'r')
    for l in cardiacData:
        l = l.split(',')
        hrList.append(int(l[0]))
        stElevList.append(int(l[1]))
        ageList.append(int(l[2]))
        prevACSList.append(int(l[3]))
        classList.append(int(l[4]))
    if toScale:
        hrList = mutils.normalize_by_std(hrList)
        stElevList = mutils.normalize_by_std(stElevList)
        ageList = mutils.normalize_by_std(ageList)
        prevACSList = mutils.normalize_by_std(prevACSList)
    #Build points
    points = []
    for i in range(len(hrList)):
        features = np.array([hrList[i], prevACSList[i],\
                            stElevList[i], ageList[i]])
        pIndex = str(i)
        points.append(Patient('P'+ pIndex, features, classList[i]))
    return points

###############################################################################

def printClustering(clustering):
    """Assumes: clustering is a sequence of clusters
       Prints information about each cluster
       Returns list of fraction of pos cases in each cluster"""
    posFracs = []
    nPts = []
    for c in clustering:
        numPts = 0
        numPos = 0
        for p in c.members():
            numPts += 1
            if p.getLabel() == 1:
                numPos += 1
        fracPos = numPos/numPts
        posFracs.append(fracPos)
        nPts.append(numPts)
        print('Cluster of size', numPts, 'with fraction of positives =',
              round(fracPos, 4))
    return nPts, posFracs

###############################################################################

def calcClustering(patients, numClusters, seed = 0, numTrials = 5):
    random.seed(seed)
    return cluster.trykmeans(patients, numClusters, numTrials)

def calcPosPatients(patients):
    numPos = 0
    for p in patients:
        if p.getLabel() == 1:
            numPos += 1
    return numPos


###############################################################################

@pytest.mark.parametrize('isScaled, k, ans_numPts, ans_fracs, ans_pos', testdata)
def test_cluster(isScaled, k, ans_numPts, ans_fracs, ans_pos):
    patients = getData(isScaled)
    scaled_str = 'SCALED' if isScaled else 'UNSCALED'
    print('\nTest ' + scaled_str + ' k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    numPts, posFracs = printClustering(bestClustering)
    assert numPts == ans_numPts
    assert posFracs == pytest.approx(ans_fracs, 0.001)

    numPos = calcPosPatients(patients)
    print('Total number of positive patients =', numPos)
    assert numPos == ans_pos
    

@pytest.mark.parametrize('k', (2,4,6))
def test_cluster_no_scale(k):
    patients = getData()
    print('\nTest UNSCALED k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    inumPts, posFracs = printClustering(bestClustering)

@pytest.mark.parametrize('k', (2,4,6))
def test_cluster_scale(k):
    patients = getData(True)
    print('\nTest SCALED k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    inumPts, posFracs = printClustering(bestClustering)

