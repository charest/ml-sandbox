from toolbox.cluster import Cluster
from toolbox.cluster import dissimilarity

import numpy as np
import random as rnd
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

###############################################################################

def kmeans(examples, k, verbose = False):
    """Get k randomly chosen initial centroids, create cluster for each"""
    initialCentroids = rnd.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(Cluster([e]))
        
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

###############################################################################

def trykmeans(examples, numClusters, numTrials, verbose = False):
    """
    Calls kmeans numTrials times and returns the result with the
    lowest dissimilarity
    """
    best = kmeans(examples, numClusters, verbose)
    minDissimilarity = dissimilarity(best)
    trial = 1
    while trial < numTrials:
        try:
            clusters = kmeans(examples, numClusters, verbose)
        except ValueError:
            continue #If failed, try again
        currDissimilarity = dissimilarity(clusters)
        if currDissimilarity < minDissimilarity:
            best = clusters
            minDissimilarity = currDissimilarity
        trial += 1
    return best

###############################################################################

def findClosestCentroids(X, centroids):
    """
    FINDCLOSESTCENTROIDS computes the centroid memberships for every example
       idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
       in idx for a dataset X where each row is a single example. idx = m x 1 
       vector of centroid assignments (i.e. each entry in range [1..K])
    """
    
    # Set K
    K = np.asarray(centroids).shape[0]
    
    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the 
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
   
    for i in range(X.shape[0]):
        dist = X[i] - centroids
        dist2 = np.sum( dist * dist, axis=1 )
        min_dist = np.argmin( dist2 )
        idx[i] = min_dist
    
    # =============================================================
   
    return idx

###############################################################################

def computeCentroids(Xin, idx):
    """
    COMPUTECENTROIDS returs the new centroids by computing the means of the 
    data points assigned to each centroid.
       centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
       computing the means of the data points assigned to each centroid. It is
       given a dataset X where each row is a single data point, a vector
       idx of centroid assignments (i.e. each entry in range [1..K]) for each
       example, and K, the number of centroids. You should return a matrix
       centroids, where each row of centroids is the mean of the data points
       assigned to it.
    """
    X = np.asarray(Xin)
    if X.ndim == 1:
        X = X[None,:]
    
    
    csr = csr_matrix((np.ones(idx.shape[0]), (idx, np.arange(idx.shape[0]))))
    
    centroids = csr*X

    counts = np.bincount(idx)
    
    return centroids/counts[:,np.newaxis]

###############################################################################

def runKMeans(X, initial_centroids, max_iters, plot_progress = None):
    """
    RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
       [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
       plot_progress) runs the K-Means algorithm on data matrix X, where each 
       row of X is a single example. It uses initial_centroids used as the
       initial centroids. max_iters specifies the total number of interactions 
       of K-Means to execute. plot_progress is a true/false flag that 
       indicates if the function should also plot its progress as the 
       learning happens. This is set to false by default. runkMeans returns 
       centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
       vector of centroid assignments (i.e. each entry in range [1..K])
    """

    # Initialize values
    previous_centroids = np.asarray(initial_centroids)
    centroids = previous_centroids
    idx = []
    K = centroids.shape[0]

    if plot_progress:
        plt.ion()
 
    # Run K-Means
    for i in range(max_iters):
        
        # Output progress
        print('K-Means iteration {}/{}...'.format(i, max_iters))
        
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
        
        # Optionally, plot progress here
        if plot_progress:
            for j in range(K):
                pos = idx==j
                plt.scatter(X[pos,0], X[pos,1])
            for j in range(K):
                plt.plot(
                    [previous_centroids[j,0], centroids[j,0]],
                    [previous_centroids[j,1], centroids[j,1]],
                    color='black', marker='X')
            previous_centroids = centroids
        
        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx)
    
    if plot_progress:
        plt.ioff()
        print("Writing to {}".format(plot_progress))
        plt.savefig(plot_progress)
        plt.close()

    return centroids, idx

###############################################################################

def initCentroids(X, K):
    """
    KMEANSINITCENTROIDS This function initializes K centroids that are to be 
    used in K-Means on the dataset X
       centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
       used with the K-Means on the dataset X
    """

    idx = np.random.choice(X.shape[0], K)
    return X[idx,...]
