import numpy as np
import random as rng

###############################################################################
def linearKernel(x1, x2):
    """
    LINEARKERNEL returns a linear kernel between x1 and x2
       sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
       and returns the value in sim
    """

    # Ensure that x1 and x2 are column vectors
    
    # Compute the kernel
    return x1 @ x2.T # dot product

###############################################################################
def gaussianKernel(X1, X2, sigma):
    """
    RBFKERNEL returns a radial basis function kernel between x1 and x2
       sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
       and returns the value in sim
    """
    x1 = X1[None,:] if X1.ndim == 1 else X1
    x2 = X2[None,:] if X2.ndim == 1 else X2
            
    x1 = np.sum(x1**2, axis=1)
    x2 = np.sum(x2**2, axis=1)
    
    K1 = x1[:, np.newaxis] + x2 - 2*X1@X2.T
    K1 = np.exp( - K1 / (2 * (sigma**2)) )

    #K = np.zeros((X1.shape[0], X2.shape[0]))
    #for i, x1 in enumerate(X1):
    #    for j, x2 in enumerate(X2):
    #        K[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (sigma**2)))

    return K1


class Model:
    def __init__(self, X, Y, kernelF, b, alphas, w):
        self.X = X
        self.y = Y
        self.kernelFunction = kernelF
        self.b = b
        self.alphas = alphas
        self.w = w

def train(X, Yin, C, kernelFunction, tol = 1e-3, max_passes = 5):
    """
    SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
    algorithm. 
       [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
       SVM classifier and returns trained model. X is the matrix of training 
       examples.  Each row is a training example, and the jth column holds the 
       jth feature.  Y is a column matrix containing 1 for positive examples 
       and 0 for negative examples.  C is the standard SVM regularization 
       parameter.  tol is a tolerance value used for determining equality of 
       floating point numbers. max_passes controls the number of iterations
       over the dataset (without changes to alpha) before the algorithm quits.
    
     Note: This is a simplified version of the SMO algorithm for training
           SVMs. In practice, if you want to train an SVM classifier, we
           recommend using an optimized package such as:  
    
               LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
               SVMLight (http://svmlight.joachims.org/)
    """
    
    
    
    # Data parameters
    m = X.shape[0]
    n = X.shape[1]
    
    # Map 0 to -1
    Y = Yin.astype(int)
    Y[Y==0] = -1
    
    # Variables
    alphas = np.zeros(m)
    b = 0
    E = np.zeros(m)
    passes = 0
    eta = 0
    L = 0
    H = 0
    
    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    #  gracefully will _not_ do this)
    # Pre-compute the Kernel Matrix
    # The following can be slow due to the lack of vectorization
    K = kernelFunction(X, X)
    
    # Train
    while passes < max_passes:
                
        num_changed_alphas = 0
        for i in range(m):
            
            # Calculate Ei = f(x[i]) - y[i] using (2). 
            # E[i] = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y[i];
            E[i] = b + np.sum(alphas*Y*K[:,i]) - Y[i]
            
            if ((Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0)):
                
                # In practice, there are many heuristics one can use to select
                # the i and j. In this simplified code, we select them randomly.
                j = rng.randint(0, m-1)
                while j == i:  # Make sure i \neq j
                    j = rng.randint(0, m-1)
    
                # Calculate Ej = f(x[j]) - y[j] using (2).
                E[j] = b + np.sum(alphas*Y*K[:,j]) - Y[j]
    
                # Save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
                
                # Compute L and H by (10) or (11). 
                if (Y[i] == Y[j]):
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
               
                if (L == H):
                    continue # continue to next i. 
    
                # Compute eta by (14).
                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                if (eta >= 0):
                    continue # continue to next i. 
                
                # Compute and clip new value for alpha j using (12) and (15).
                alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta
                
                # Clip
                alphas[j] = min (H, alphas[j])
                alphas[j] = max (L, alphas[j])
                
                # Check if change in alpha is significant
                if (abs(alphas[j] - alpha_j_old) < tol):
                    # continue to next i. 
                    # replace anyway
                    alphas[j] = alpha_j_old
                    continue
                
                # Determine value for alpha i using (16). 
                alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j])
                
                # Compute b1 and b2 using (17) and (18) respectively. 
                b1 = b - E[i] \
                    - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j] \
                    - Y[j] * (alphas[j] - alpha_j_old) *  K[i,j]
                b2 = b - E[j] \
                    - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j] \
                    - Y[j] * (alphas[j] - alpha_j_old) *  K[j,j]
    
                # Compute b by (19). 
                if (0 < alphas[i] and alphas[i] < C):
                    b = b1;
                elif (0 < alphas[j] and alphas[j] < C):
                    b = b2;
                else:
                    b = (b1+b2)/2
    
                num_changed_alphas = num_changed_alphas + 1
        
        if (num_changed_alphas == 0):
            passes = passes + 1
        else:
            passes = 0

    # Save the model
    idx = alphas > 0
    model = Model( X[idx,:], Y[idx], kernelFunction, b, alphas[idx], ((alphas*Y) @ X) )

    return model

def predict(model, Xin, precompute=False):
    """
    SVMPREDICT returns a vector of predictions using a trained SVM model
    (svmTrain). 
       pred = SVMPREDICT(model, X) returns a vector of predictions using a 
       trained SVM model (svmTrain). X is a mxn matrix where there each 
       example is a row. model is a svm model returned from svmTrain.
       predictions pred is a m x 1 column of predictions of {0, 1} values.
    """
    
    # Check if we are getting a column vector, if so, then assume that we only
    # need to do prediction for a single example
    if Xin.ndim == 1:
        X = Xin.reshape(1, -1)
    else:
        X = Xin
    
    # Dataset 
    m = X.shape[0]
    
    if precompute:
        K = model.kernelFunction( X, model.X )
        K = model.y * model.alphas * K
        p = np.sum(K, axis=1) + model.b
    else:
        p = np.zeros(m)
        # Other Non-linear kernel
        for i in range(m):
            prediction = 0
            xi = X[i,:].reshape(1,-1)
            for j in range(model.X.shape[0]):
                xj = model.X[j,:].reshape(1,-1)
                prediction = prediction + \
                    model.alphas[j] * model.y[j] * \
                    model.kernelFunction(xi, xj)
            p[i] = prediction + model.b

    # Convert predictions into 0 / 1
    pred = np.zeros(m)
    pred[p >= 0] =  1
    pred[p <  0] =  0

    return pred.astype(int)
