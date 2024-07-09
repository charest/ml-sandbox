import numpy as np

def estimateGaussian(X):
    """
    ESTIMATEGAUSSIAN This function estimates the parameters of a 
    Gaussian distribution using the data in X
       [mu sigma2] = estimateGaussian(X), 
       The input X is the dataset with each n-dimensional data point in one row
       The output is an n-dimensional vector mu, the mean of the data set
       and the variances sigma^2, an n x 1 vector
    """
    
    m, n = X.shape
    
    mu = np.sum(X, axis=0) / m
    delta = X - mu
    sigma2 = np.sum( delta*delta , axis=0) / m

    return mu, sigma2

def multivariateGaussian(X, mu, Sigma2):
    """
    MULTIVARIATEGAUSSIAN Computes the probability density function of the
    multivariate gaussian distribution.
        p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
        density function of the examples X under the multivariate gaussian 
        distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
        treated as the covariance matrix. If Sigma2 is a vector, it is treated
        as the sigma^2 values of the variances in each dimension (a diagonal
        covariance matrix)
    """
    
    k = len(mu)
    
    if Sigma2.ndim == 1 or Sigma2.shape[0] == 1 or Sigma2.shape[1] == 1:
        Var = np.diag(Sigma2)
    else:
        Var = Sigma2
    
    det = np.linalg.det(Var)
    Xnorm = X - mu
    tmp1 = Xnorm @ np.linalg.pinv(Var)
    tmp2 = tmp1 * Xnorm
    tmp3 = np.sum(tmp2, axis=1)

    p = (2 * np.pi) ** (- k / 2) * det ** (-0.5) * np.exp(-0.5 * tmp3) 

    return p
    

def selectThreshold(yval, pval, eps=1.e-9):
    """
    SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
    outliers
       [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
       threshold to use for selecting outliers based on the results from a
       validation set (pval) and the ground truth (yval).
    """
    
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
   
    maxp = np.max(pval)
    minp = np.min(pval)

    stepsize = (maxp - minp) / 1000
    for epsilon in np.arange(minp, maxp, stepsize):

        pred = pval < epsilon
        tru = pred == yval
        fal = np.logical_not(tru)
        
        tp = np.sum( pred[tru] == 1 )
        fp = np.sum( pred[fal] == 1 )
        fn = np.sum( pred[fal] == 0 )

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)

        F1 = 2 * prec * rec / (prec + rec + eps)

        
    
        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon

    return bestEpsilon, bestF1


