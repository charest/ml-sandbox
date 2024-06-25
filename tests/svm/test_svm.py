
import pytest

import toolbox.file_utils as futils
import toolbox.svm as svm

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

import numpy as np
from scipy.optimize import minimize
import scipy.io as sio

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

def plotData(X, y, fname = None):

    pos = y==1
    neg = y==0
    plt.scatter(X[pos,0], X[pos, 1], marker='x', color='black', label='Positive')
    plt.scatter(X[neg,0], X[neg, 1], marker='o', color='black', label='Negative')
    plt.legend()
    if fname == None:
        plt.show()
    else:
        print('Saving figure to {}'.format(fname))
        plt.savefig(fname)

    plt.close()

###############################################################################

def visualizeBoundaryLinear(X, y, model, fname = None):
    """
    VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    SVM
       VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
       learned by the SVM and overlays the data on it
    """
    
    w = model.w
    b = model.b
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = - (w[0]*xp + b)/w[1]
    
    pos = y==1
    neg = y==0
    plt.scatter(X[pos,0], X[pos, 1], marker='x', color='black', label='Positive')
    plt.scatter(X[neg,0], X[neg, 1], marker='o', color='black', label='Negative')
    
    plt.plot(xp, yp, color='blue') 
    
    plt.legend()
    
    if fname == None:
        plt.show()
    else:
        print('Saving figure to {}'.format(fname))
        plt.savefig(fname)

    plt.close()

###############################################################################

def visualizeBoundary(X, y, model, fname = None):
    """
    VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
       VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
       boundary learned by the SVM and overlays the data on it
    """
    
    w = model.w;
    b = model.b;
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)

    vals = np.zeros(X1.shape)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            this_X = np.array([ X1[i,j], X2[i,j] ])
            vals[i,j] = svm.predict(model, this_X)
    #print(np.sum(vals==0), np.sum(vals==1))
    
    pos = y==1
    neg = y==0
    plt.scatter(X[pos,0], X[pos, 1], marker='x', color='black', label='Positive')
    plt.scatter(X[neg,0], X[neg, 1], marker='o', color='black', label='Negative')
    
    plt.contour(X1, X2, vals)
    
    plt.legend()
    
    if fname == None:
        plt.show()
    else:
        print('Saving figure to {}'.format(fname))
        plt.savefig(fname)

    plt.close()

###############################################################################

def datasetParams(X, y, Xval, yval):
    """
    EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel
       [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
       sigma. You should complete this function to return the optimal C and 
       sigma based on a cross-validation set.
    """

    vals = {}

    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
    
            print("Training C={:.2f} sigma={:.2f}".format(C,sigma), end='')
            k = lambda x1,x2 : svm.gaussianKernel(x1, x2, sigma)
            model = svm.train(X, y, C, k)
            
            ytest = svm.predict(model, Xval, True)
            acc = np.mean(ytest != yval)
            vals[(C,sigma)] = acc
            print(" err = {:.6f}".format(acc))
            #print("ytest ", np.sum(ytest==0), np.sum(ytest==1))
            #print("yval  ", np.sum(yval==0), np.sum(yval==1))
            #print("wrong ", np.sum(ytest != yval))
            #print("right ", np.sum(ytest == yval))

    C,sigma = min(vals, key=vals.get)
    print("Picking C={:.2f} sigma={:.2f} err={:.2f}".format(C,sigma,vals[(C,sigma)]))

    return C, sigma

###############################################################################

def test_linear_kernel():
    
    # =============== Part 3: Implementing Gaussian Kernel ===============
    # You will now implement the Gaussian kernel to use
    # with the SVM. You should complete the code in gaussianKernel.m
    print('Evaluating the linear Kernel ...')
    
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = svm.linearKernel(x1, x2)
    
    assert sim == pytest.approx(7, abs=1.e-6)

    
###############################################################################

def test_gaussian_kernel():
    
    # =============== Part 3: Implementing Gaussian Kernel ===============
    # You will now implement the Gaussian kernel to use
    # with the SVM. You should complete the code in gaussianKernel.m
    print('Evaluating the Gaussian Kernel ...')
    
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = svm.gaussianKernel(x1, x2, sigma);
    
    print('Gaussian Kernel between x1 = {}, x2 = {}, sigma = {} : {}'.format(x1, x2, sigma, sim))
    print('(this value should be about 0.324652)')
    assert sim == pytest.approx(0.324652, abs=1.e-6)

###############################################################################

def test_svm_linear(tmp_path):
    
    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # The following code will load the dataset into your environment and plot
    # the data.
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex6data1.mat")
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    print("\nData keys:")
    print("  ", list(data.keys()))
    
    X = data['X']
    y = data['y']
    
    assert X.ndim == 2
    assert X.shape == (51, 2)
    
    assert y.ndim == 2
    assert y.shape == (51,1)

    y = y.ravel()

    # number of training examples
    m = X.shape[0]
    n = X.shape[1]
    
    # Plot training data
    plotData(X, y, tmp_path/"ex6data1.png")
    #plotData(X, y)

    # ==================== Part 2: Training Linear SVM ====================
    # The following code will train a linear SVM on the dataset and plot the
    # decision boundary learned.
    
    print('Training Linear SVM ...')
    
    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1
    model = svm.train(X, y, C, svm.linearKernel, 1e-3, 20)
    visualizeBoundaryLinear(X, y, model, tmp_path/"ex6data1-bnd-c{:04d}.png".format(C))

    C = 1000
    model = svm.train(X, y, C, svm.linearKernel, 1e-3, 20)
    visualizeBoundaryLinear(X, y, model, tmp_path/"ex6data1-bnd-c{:04d}.png".format(C))

###############################################################################

def test_svm_rbf(tmp_path):
    
    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # The following code will load the dataset into your environment and plot
    # the data.
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex6data2.mat")
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    print("\nData keys:")
    print("  ", list(data.keys()))
    
    X = data['X']
    y = data['y']
    
    assert X.ndim == 2
    assert X.shape == (863, 2)
    
    assert y.ndim == 2
    assert y.shape == (863,1)

    y = y.ravel()

    # number of training examples
    m = X.shape[0]
    n = X.shape[1]
    
    # Plot training data
    plotData(X, y, tmp_path/"ex6data2.png")

    # ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    # After you have implemented the kernel, we can now use it to train the 
    # SVM classifier.
    
    print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')
    
    # SVM Parameters
    C = 1
    sigma = 0.1
    
    # We set the tolerance and max_passes lower here so that the code will run
    # faster. However, in practice, you will want to run the training to
    # convergence.
    k = lambda x1,x2 : svm.gaussianKernel(x1, x2, sigma)
    model = svm.train(X, y, C, k)
    visualizeBoundary(X, y, model, tmp_path/"ex6data2-bnd-c{:04d}.png".format(C))

###############################################################################

def test_svm_rbf2(tmp_path):
    
    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # The following code will load the dataset into your environment and plot
    # the data.
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex6data3.mat")
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    print("\nData keys:")
    print("  ", list(data.keys()))
    
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    
    assert X.ndim == 2
    assert X.shape == (211, 2)
    
    assert y.ndim == 2
    assert y.shape == (211,1)

    y = y.ravel()
    yval = yval.ravel()

    # number of training examples
    m = X.shape[0]
    n = X.shape[1]
    
    # Plot training data
    plotData(X, y, tmp_path/"ex6data3.png")

    # ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
    #
    # This is a different dataset that you can use to experiment with. Try
    # different values of C and sigma here.
    
    print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')
    
    # SVM Parameters
    C, sigma = datasetParams(X, y, Xval, yval)
    #C = 0.1
    #sigma = 0.03

    # We set the tolerance and max_passes lower here so that the code will run
    # faster. However, in practice, you will want to run the training to
    # convergence.
    k = lambda x1,x2 : svm.gaussianKernel(x1, x2, sigma)
    model = svm.train(X, y, C, k)
    visualizeBoundary(X, y, model, tmp_path/"ex6data3-bnd.png")

