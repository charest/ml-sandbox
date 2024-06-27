import pytest

from toolbox.cluster import Example
import toolbox.cluster.kmeans as kmeans
import toolbox.file_utils as futils
import toolbox.math_utils as mutils

import random as rnd
import numpy as np

import scipy.io as sio
import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

testdata = [
    (False, 2, [118, 132], [0.3305, 0.3333], 83),
    (True,  2, [224,  26], [0.2902, 0.6923], 83),
]


class Patient(Example):
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
    rnd.seed(seed)
    return kmeans.trykmeans(patients, numClusters, numTrials)

def calcPosPatients(patients):
    numPos = 0
    for p in patients:
        if p.getLabel() == 1:
            numPos += 1
    return numPos


###############################################################################

@pytest.mark.parametrize('isScaled, k, ans_numPts, ans_fracs, ans_pos', testdata)
def test_kmeans(isScaled, k, ans_numPts, ans_fracs, ans_pos):
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
def test_kmeans_no_scale(k):
    patients = getData()
    print('\nTest UNSCALED k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    inumPts, posFracs = printClustering(bestClustering)

@pytest.mark.parametrize('k', (2,4,6))
def test_kmeans_scale(k):
    patients = getData(True)
    print('\nTest SCALED k-means (k = ' + str(k) + ')')
    bestClustering = calcClustering(patients, k, 2)
    inumPts, posFracs = printClustering(bestClustering)

###############################################################################

def test_kmeans_2(tmp_path):
    
    # ================= Part 1: Find Closest Centroids ====================
    # To help you implement K-Means, we have divided the learning algorithm 
    # into two functions -- findClosestCentroids and computeCentroids. In this
    # part, you shoudl complete the code in the findClosestCentroids function. 
    print('\nFinding closest centroids.')
    
    # Load an example dataset that we will be using
    data = sio.loadmat(TEST_DATA_DIR/"ex7data2.mat")
    assert 'X' in data.keys()

    X = data["X"]
   
    # Select an initial set of centroids
    K = 3 # 3 Centroids
    initial_centroids = [[3, 3], [6, 2], [8, 5]]
    
    # Find the closest centroids for the examples using the
    # initial_centroids
    idx = kmeans.findClosestCentroids(X, initial_centroids)
    
    print('Closest centroids for the first 3 examples: {}'.format(idx[0:3]))
    print('(the closest centroids should be 0, 2, 1 respectively)')

    assert (idx[0:3] == [0, 2, 1]).all()

    # ===================== Part 2: Compute Means =========================
    # After implementing the closest centroids function, you should now
    # complete the computeCentroids function.
    print('Computing centroids means.')
    
    # Compute means based on the closest centroids found in the previous part.
    centroids = kmeans.computeCentroids(X, idx)
    
    print('Centroids computed after initial finding of closest centroids:')
    print('{}'.format(centroids))
    print('(the centroids should be')
    print(' [ 2.428301 3.157924 ]')
    print(' [ 5.813503 2.633656 ]')
    print(' [ 7.119387 3.616684 ]')
    
    ans = np.array(
        [[ 2.428301, 3.157924 ],
         [ 5.813503, 2.633656 ],
         [ 7.119387, 3.616684 ]])
    np.testing.assert_allclose(centroids.ravel(), ans.ravel(), atol=1.e-6)

    # =================== Part 3: K-Means Clustering ======================
    # After you have completed the two functions computeCentroids and
    # findClosestCentroids, you have all the necessary pieces to run the
    # kMeans algorithm. In this part, you will run the K-Means algorithm on
    # the example dataset we have provided. 
    print('Running K-Means clustering on example dataset.')
    
    # Settings for running K-Means
    max_iters = 10
    
    # For consistency, here we set centroids to specific values
    # but in practice you want to generate them automatically, such as by
    # settings them to be random examples (as can be seen in
    # kMeansInitCentroids).
    initial_centroids = [[3, 3], [6, 2], [8, 5]]
    
    # Run K-Means algorithm. The 'true' at the end tells our function to plot
    # the progress of K-Means
    centroids, idx = kmeans.runKMeans(X, initial_centroids, max_iters, tmp_path/"ex7data2-clusters.png")
    print('K-Means Done.')
    np.testing.assert_allclose(centroids,
        [[1.95399466, 5.02557006],
         [3.04367119, 1.01541041],
         [6.03366736, 3.00052511]], atol=1.e-6)

###############################################################################

def test_kmeans_png(tmp_path, BirdImage):
    

    # ============= Part 4: K-Means Clustering on Pixels ===============
    # In this exercise, you will use K-Means to compress an image. To do this,
    # you will first run K-Means on the colors of the pixels in the image and
    # then you will map each pixel on to it's closest centroid.
    
    print('Running K-Means clustering on pixels from an image.');
    
    #  Load an image of a bird
    A = BirdImage

    A = A / 255 # Divide by 255 so that all values are in the range 0 - 1
    
    # Size of the image
    img_size = A.shape
    
    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = np.reshape(A, (img_size[0] * img_size[1], 3))
    
    # Run your K-Means algorithm on this data
    # You should try different values of K and max_iters here
    K = 16
    max_iters = 10
    
    # When using K-Means, it is important the initialize the centroids
    # randomly. 
    # You should complete the code in kMeansInitCentroids.m before proceeding
    rnd.seed(0)
    initial_centroids = kmeans.initCentroids(X, K)
    
    # Run K-Means
    centroids, idx = kmeans.runKMeans(X, initial_centroids, max_iters)

    # ================= Part 5: Image Compression ======================
    # In this part of the exercise, you will use the clusters of K-Means to
    # compress an image. To do this, we first find the closest clusters for
    # each example. After that, we 
    
    print('Applying K-Means to compress an image.');
    
    # Find closest cluster members
    idx = kmeans.findClosestCentroids(X, centroids)
    
    # Essentially, now we have represented the image X as in terms of the
    # indices in idx. 
    
    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by it's index in idx) to the centroid value
    X_recovered = centroids[idx,:]
    
    # Reshape the recovered image into proper dimensions
    X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1], 3))
    
    # Display the original image 
    plt.subplot(1, 2, 1)
    plt.imshow(A*255, extent=[0, 1, 0, 1])
    plt.title('Original');
    
    # Display compressed image side by side
    plt.subplot(1, 2, 2)
    plt.imshow(X_recovered*255, extent=[0, 1, 0, 1])
    plt.title('Compressed, with {} colors.'.format(K))

    fname = tmp_path / "bird_small-compress.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)
