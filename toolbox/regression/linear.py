import numpy as np

try:
    from collections.abc import Sized
except ImportError:
    from collections import Sized

def computeCost(X, y, theta):
    res = X @ theta
    res -= y
    return 0.5 * np.dot( res, res )/len(X)

def costFunction(X, y, theta, lmda=None):
    assert X.shape[0] == y.shape[0]
    m = X.shape[0]
    res = X @ theta - y
    thetaR = theta[1:]
    J = 0.5 * np.dot( res, res )
    grad = res @ X
    if lmda != None:
        J += 0.5 * lmda * np.dot(thetaR, thetaR)
        grad[1:] += lmda * thetaR
    return J/m, grad/m


def gradientDescent(X, y, theta, alpha, iterations, freq=100):
    J_hist = []
    if (freq>0):
        print("{:>6}\t{:>8}".format("its", "cost"))
    for i in range(iterations):
        predictions = X @ theta
        error = predictions - y
        grad = ( X.T @ error ) / len(X)
        theta = theta - alpha * grad
        J = 0.5 * np.dot( error, error ) / len(X)
        J_hist.append(J)
        if (freq>0 and (i) % freq == 0):
            print("{:6d}\t{:8.2e}".format(i, J))
    return theta, J_hist

def normalEqn(X, y):
    A = X.T @ X 
    b = X.T @ y
    theta = np.linalg.solve(A, b)
    return theta
    
 
def featureNormalize(X, mu=None, sigma=None):
    """Normalize so that mean is zero and standard deviation is 1"""
    mu_out = mu
    sigma_out = sigma
    if mu_out is None:
        mu_out = np.sum(X, axis=0) / X.shape[0]
    if sigma_out is None:
        sigma_out = np.std(X, axis=0)
    Xnorm = X - mu_out
    return Xnorm/sigma_out, mu_out, sigma_out

def mapPolyFeatures(X, degree=6):
    """
    POLYFEATURES Maps X (1D vector) into the p-th power
       [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
       maps each example into its polynomial features where
       X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    """
    
    m = len(X) if isinstance(X, Sized) else 1
    out = X
    
    for i in range(1,degree):
        out = np.column_stack((out, X**(i+1)))


    return out

