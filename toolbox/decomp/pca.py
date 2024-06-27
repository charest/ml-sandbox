import numpy as np
from numpy.linalg import svd
from scipy.linalg import diagsvd

def featureNormalize(X):
    """Normalize so that mean is zero and standard deviation is 1"""
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    # Set Delta Degrees of Freedom (ddof) to 1, to compute
    # the std based on a sample and not the population
    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

def pca(X):
    """
    PCA Run principal component analysis on the dataset X
       [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
       Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """

    m, n = X.shape
    cov= X.T @ X / m

    U, S, Vh = svd(cov)
    #sigma = diagsvd(S, len(S), len(S))
    return U, S#, sigma

def projectData(X, U, K):
    """
    PROJECTDATA Computes the reduced data representation when projecting only 
    on to the top k eigenvectors
       Z = projectData(X, U, K) computes the projection of 
       the normalized inputs X into the reduced dimensional space spanned by
       the first K columns of U. It returns the projected examples in Z.
    """
    
    #projection_k = x.T @ U[:, k]
    return X @ U[:,0:K]

def recoverData(Z, U, K):
    """
    RECOVERDATA Recovers an approximation of the original data when using the 
    projected data
       X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
       original data that has been reduced to K dimensions. It returns the
       approximate reconstruction in X_rec.
    """
    return Z @ U[:,0:K].T
