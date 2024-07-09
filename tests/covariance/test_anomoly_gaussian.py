import pytest

import toolbox.decomp.pca as pca
import toolbox.file_utils as futils
from  toolbox.covariance.gaussian import *

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

def visualizeFit(X, mu, sigma2):
    """
    VISUALIZEFIT Visualize the dataset and its estimated distribution.
       VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
       probability density function of the Gaussian distribution. Each example
       has a location (x1, x2) that depends on its feature values.
    """

    xs = np.arange(0, 35, 0.5)
    X1,X2 = np.meshgrid(xs, xs) 
    Z = multivariateGaussian(np.column_stack((X1.reshape(X1.size),X2.reshape(X2.size))),mu,sigma2)
    Z = Z.reshape(X1.shape)
    
    #plt.scatter(X[:,0], X[:,1])
    
    # Do not plot if there are infinities
    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, np.power(10,(np.arange(-20, 0.1, 3)).T))

###############################################################################

def test_anomoly_gaussian(tmp_path):
    
    # ================== Part 1: Load Example Dataset  ===================
    # We start this exercise by using a small dataset that is easy to
    # visualize.
    #
    # Our example case consists of 2 network server statistics across
    # several machines: the latency and throughput of each machine.
    # This exercise will help us find possibly faulty (or very fast) machines.
    
    print('\nVisualizing example dataset for outlier detection.')
    
    # The following command loads the dataset. You should now have the
    # variables X, Xval, yval in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex8data1.mat")
    assert 'X' in data.keys()
    assert 'Xval' in data.keys()
    assert 'yval' in data.keys()

    X = data["X"]
    Xval = data["Xval"]
    yval = data["yval"].ravel()
    
    # Visualize the example dataset
    plt.scatter(X[:, 0], X[:, 1], marker='x', color='blue')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    
    fname = tmp_path / "ex8data1.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)

    # ================== Part 2: Estimate the dataset statistics ===================
    # For this exercise, we assume a Gaussian distribution for the dataset.
    #
    # We first estimate the parameters of our assumed Gaussian distribution, 
    # then compute the probabilities for each of the points and then visualize 
    # both the overall distribution and where each of the points falls in 
    # terms of that distribution.
    print('Visualizing Gaussian fit.')
    
    # Estimate my and sigma2
    mu, sigma2 = estimateGaussian(X)
    # Returns the density of the multivariate normal at each data point (row) 
    # of X
    p = multivariateGaussian(X, mu, sigma2)
    
    # Visualize the fit
    visualizeFit(X,  mu, sigma2)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')

    outfile = tmp_path / "ex8data1-threshold.png"
    print('Writing output to {}'.format(outfile))
    plt.savefig(outfile)

    # ================== Part 3: Find Outliers ===================
    # Now you will find a good epsilon threshold using a cross-validation set
    # probabilities given the estimated Gaussian distribution
    
    pval = multivariateGaussian(Xval, mu, sigma2)
    
    epsilon, F1 = selectThreshold(yval, pval)
    print('Best epsilon found using cross-validation: {}'.format(epsilon))
    print('Best F1 on Cross Validation Set:  {}'.format(F1))
    print('   (you should see a value epsilon of about 8.99e-05)')
    
    assert epsilon == pytest.approx(8.99e-5, rel=1.e-4)

    
    # Find the outliers in the training set and plot the
    outliers = np.where(p < epsilon)
    
    # Draw a red circle around those outliers
    plt.scatter(X[outliers,0], X[outliers,1], facecolors='none', edgecolors='r', s=100)
    
    outfile = tmp_path / "ex8data1-outliers.png"
    print('Writing output to {}'.format(outfile))
    plt.savefig(outfile)

###############################################################################

def test_anomoly_gaussian_multi(tmp_path):
    
    # ================== Part 1: Load Example Dataset  ===================
    # We will now use the code from the previous part and apply it to a 
    # harder problem in which more features describe each datapoint and only 
    # some features indicate whether a point is an outlier.
    
    # The following command loads the dataset. You should now have the
    # variables X, Xval, yval in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex8data2.mat")
    assert 'X' in data.keys()
    assert 'Xval' in data.keys()
    assert 'yval' in data.keys()

    X = data["X"]
    Xval = data["Xval"]
    yval = data["yval"].ravel()

    # ================== Part 4: Multidimensional Outliers ===================
    
    # Apply the same steps to the larger dataset
    mu, sigma2 = estimateGaussian(X)
    
    # Training set 
    p = multivariateGaussian(X, mu, sigma2)
    
    # Cross-validation set
    pval = multivariateGaussian(Xval, mu, sigma2)
    
    # Find the best threshold
    epsilon, F1 = selectThreshold(yval, pval)
    
    nout = np.sum(p < epsilon)
    
    print('Best epsilon found using cross-validation: {}'.format(epsilon))
    print('Best F1 on Cross Validation Set:  {}'.format(F1))
    print('# Outliers found: {}'.format(nout))
    print('   (you should see a value epsilon of about 1.38e-18)')
    
    assert nout == 117
    assert epsilon == pytest.approx(1.38e-18, abs=1.e-20)

