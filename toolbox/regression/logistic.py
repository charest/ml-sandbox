import collections
import numpy as np

try:
    from collections.abc import Sized
except ImportError:
    from collections import Sized


def sigmoid(x):
    e = np.exp(-x)
    return 1 / (1 + e)

def sigmoidGradient(x):
    s = sigmoid(x)
    return s * (1 - s)

#def costFunction(theta, X, y):
#    h = sigmoid( X @ theta.T )
#    one_m_y = 1 - y
#    one_m_h = 1 - h
#    res = - y @ np.log(h) - one_m_y @ np.log(one_m_h)
#    J = res / len(X)
#    return J

def costFunctionGrad(theta, X, y):
    h = sigmoid( X @ theta )
    error = h - y
    grad = ( error @ X ) / len(X)
    return grad

def predict(theta, X):
    prob = np.round( sigmoid(X @ theta.T) )
    return prob

def predictOneVsAll(theta, X):
    prob = sigmoid( X @ theta.T )
    imax = np.argmax(prob, axis=1)
    return imax

def costFunction(theta, X, y, lmda = None):
    h = sigmoid( X @ theta.T )
    one_m_y = 1 - y
    one_m_h = 1 - h
    J = - y.T @ np.log(h) - one_m_y.T @ np.log(one_m_h)
    error = h - y
    grad = ( error.T @ X )
    if lmda != None:
        thetaR = theta[1:]
        J += lmda/2 * thetaR @ thetaR.T
        grad[1:] += lmda*thetaR
    m = len(X)
    return J/m, grad/m

def costFunctionGradReg(theta, X, y, lmda):
    h = sigmoid( X @ theta.T )
    error = h - y
    grad = ( error.T @ X )
    if theta.ndim == 1:
        grad[1:] += lmda*theta[1:]
    else:
        grad[:,1:] += lmda*theta[:,1:]
        raise ValueError('multi value cost functions not tested.')
    return grad / len(X)

def mapPolyFeature(X1, X2, degree=6):
    """
    MAPFEATURE Feature mapping function to polynomial features
    
      MAPFEATURE(X1, X2) maps the two input features
      to quadratic features used in the regularization exercise.
    
      Returns a new feature array with more features, comprising of 
      X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    
      Inputs X1, X2 must be the same size
    """
    
    m = len(X1) if isinstance(X1, Sized) else 1
    out = np.ones(m) # Add a column of ones to x
    
    for i in range(1,degree+1):
        for j in range(i+1):
            out = np.c_[out, (X1**(i-j))*(X2**j)]


    return out
