import pytest
import random, pylab, numpy
import toolbox.cluster as cluster

from pathlib import Path

TEST_DATA_DIR = Path(__file__).resolve().parent

testdata = [
    (False, 2, [118, 132], [0.3305, 0.3333]),
    (True,  2, [224,  26], [0.2902, 0.6923]),
]


class Patient(cluster.Example):
    pass

def scaleAttrs(vals):
    vals = pylab.array(vals)
    mean = sum(vals)/len(vals)
    sd = numpy.std(vals)
    vals = vals - mean
    return vals/sd

def getData(toScale = False):
    #read in data
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
        hrList = scaleAttrs(hrList)
        stElevList = scaleAttrs(stElevList)
        ageList = scaleAttrs(ageList)
        prevACSList = scaleAttrs(prevACSList)
    #Build points
    points = []
    for i in range(len(hrList)):
        features = pylab.array([hrList[i], prevACSList[i],\
                                stElevList[i], ageList[i]])
        pIndex = str(i)
        points.append(Patient('P'+ pIndex, features, classList[i]))
    return points
    
def kmeans(examples, k, verbose = False):
    #Get k randomly chosen initial centroids, create cluster for each
    initialCentroids = random.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(cluster.Cluster([e]))
        
    #Iterate until centroids do not change
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        #Create a list containing k distinct empty lists
        newClusters = []
        for i in range(k):
            newClusters.append([])
            
        #Associate each example with closest centroid
        for e in examples:
            #Find the centroid closest to e
            smallestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add e to the list of examples for appropriate cluster
            newClusters[index].append(e)
            
        for c in newClusters: #Avoid having empty clusters
            if len(c) == 0:
                raise ValueError('Empty Cluster')
        
        #Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(newClusters[i]) > 0.0:
                converged = False
        if verbose:
            print('Iteration #' + str(numIterations))
            for c in clusters:
                print(c)
            print('') #add blank line
    return clusters

def trykmeans(examples, numClusters, numTrials, verbose = False):
    """Calls kmeans numTrials times and returns the result with the
          lowest dissimilarity"""
    best = kmeans(examples, numClusters, verbose)
    minDissimilarity = cluster.dissimilarity(best)
    trial = 1
    while trial < numTrials:
        try:
            clusters = kmeans(examples, numClusters, verbose)
        except ValueError:
            continue #If failed, try again
        currDissimilarity = cluster.dissimilarity(clusters)
        if currDissimilarity < minDissimilarity:
            best = clusters
            minDissimilarity = currDissimilarity
        trial += 1
    return best

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

def calcClustering(patients, numClusters, seed = 0, numTrials = 5):
    random.seed(seed)
    return trykmeans(patients, numClusters, numTrials)

@pytest.mark.parametrize('isScaled, k, ans_numPts, ans_fracs', testdata)
def test_cluster(isScaled, k, ans_numPts, ans_fracs):
    patients = getData(isScaled)
    scaled_str = 'SCALED' if isScaled else 'UNSCALED'
    print('Test ' + scaled_str + ' k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    numPts, posFracs = printClustering(bestClustering)
    assert numPts == ans_numPts
    assert posFracs == pytest.approx(ans_fracs, 0.001)
    

@pytest.mark.parametrize('k', (2,4,6))
def test_cluster_no_scale(k):
    patients = getData()
    print('Test UNSCALED k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    inumPts, posFracs = printClustering(bestClustering)

@pytest.mark.parametrize('k', (2,4,6))
def test_cluster_scale(k):
    patients = getData(True)
    print('Test SCALED k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    inumPts, posFracs = printClustering(bestClustering)

#numPos = 0
#for p in patients:
#    if p.getLabel() == 1:
#        numPos += 1
#print('Total number of positive patients =', numPos)
