import pytest

import toolbox.file_utils as futils
import toolbox.regression.linear as lr

import numpy as np
from scipy.optimize import minimize

import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

def trainLinearReg(X, y, lmda, disp=True):
    """
    TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    regularization parameter lambda
       [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
       the dataset (X, y) and regularization parameter lambda. Returns the
       trained parameters theta.
    """
    
    
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])
    
    # Create "short hand" for the cost function to be minimized
    f = lambda t : lr.costFunction(X, y, t, lmda)
    
    # Now, costFunction is a function that takes in only one argument
    opts = {
        'maxiter' : 50,
        'disp' : disp
    }
    
    # Minimize using fmincg
    res = minimize(f, initial_theta, jac=True, options=opts, method='CG')

    return res['x']

###############################################################################

def learningCurve(X, y, Xval, yval, lmda):
    """
    LEARNINGCURVE Generates the train and cross validation set errors needed 
    to plot a learning curve
       [error_train, error_val] = ...
           LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
           cross validation set errors for a learning curve. In particular, 
           it returns two vectors of the same length - error_train and 
           error_val. Then, error_train(i) contains the training error for
           i examples (and similarly for error_val(i)).
    
       In this function, you will compute the train and test errors for
       dataset sizes from 1 up to m. In practice, when working with larger
       datasets, you might want to do this in larger intervals.
    """

    m = X.shape[0]

    error_train = []
    error_val = []
        
    print('# Training Examples\tTrain Error\tCross Validation Error')
    
    for i in range(1,m+1):
        Xtrain = X[:i,:]
        ytrain = y[:i]
        Xtest = X[i:,:]
        ytest = y[i:]

        theta = trainLinearReg(Xtrain, ytrain, lmda, False)
        Jtrain, grad = lr.costFunction(Xtrain, ytrain, theta)
        Jval,   grad = lr.costFunction(X, y, theta)
        
        error_train.append(Jtrain)
        error_val.append(Jval)
    
        print('  \t{}\t\t{}\t{}'.format(i, Jtrain, Jval))

    return error_train, error_val


###############################################################################

def test_regression_learn(tmp_path):

    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # The following code will load the dataset into your environment and plot
    # the data.
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex5data1.mat")
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    print("\nData keys:")
    print("  ", list(data.keys()))
    
    X = data['X']
    y = data['y']
    
    Xval = data['Xval']
    yval = data['yval']

    Xtest = data['Xtest']

    assert X.ndim == 2
    assert X.shape == (12, 1)
    
    assert y.ndim == 2
    assert y.shape == (12,1)

    y = y.ravel()
    Xfull = np.insert(X, 0, 1, axis=1)

    # number of training examples
    m = X.shape[0]
    n = X.shape[1]
    
    # Plot training data
    plt.scatter(X, y, marker='x', color='black');
    plt.xlabel('Change in water level (x)');
    plt.ylabel('Water flowing out of the dam (y)');
    #plt.show()

    # =========== Part 3: Regularized Linear Regression Gradient =============
    # You should now implement the gradient for regularized linear 
    # regression.
    
    theta = np.array([1 , 1])
    J, grad = lr.costFunction(Xfull, y, theta, 1)
    
    print('Gradient at theta = [1 ; 1]:  {}] '.format(grad))
    print('(this value should be about [-15.303016; 598.250744])')
    
    np.testing.assert_allclose(grad, [-15.303016, 598.250744], atol=1e-6)
    
    # =========== Part 4: Train Linear Regression =============
    # Once you have implemented the cost and gradient correctly, the
    # trainLinearReg function will use your cost function to train 
    # regularized linear regression.
    #
    # Write Up Note: The data is non-linear, so this will not give a great 
    #                fit.
    
    # Train linear regression with lambda = 0
    lmda = 0
    theta = trainLinearReg(Xfull, y, lmda)
    
    # Plot fit over the data
    plt.plot(X, Xfull@theta)
    
    outfile = tmp_path/"ex5data1.png"
    print("Saving figure to {}".format(outfile))
    plt.savefig(outfile)
    
    #plt.show()
    plt.close()

    
    # =========== Part 5: Learning Curve for Linear Regression =============
    # Next, you should implement the learningCurve function. 
    #
    # Write Up Note: Since the model is underfitting the data, we expect to
    #                see a graph with "high bias" -- slide 8 in ML-advice.pdf 
    
    lmda = 0
    error_train, error_val = learningCurve(Xfull, y, np.insert(Xval,0,1,axis=1), yval, lmda)
    
    ids = [i for i in range(len(error_train))]
    plt.plot(ids, error_train, label='Train')
    plt.plot(ids, error_val, label='Cross Validation')
    plt.title('Learning curve for linear regression')
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.legend()
    
    outfile = tmp_path/"ex5data1-learning.png"
    print("Saving figure to {}".format(outfile))
    plt.savefig(outfile)
    
    #plt.show()
    plt.close()

###############################################################################

def test_regression_poly(tmp_path):
    
    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # The following code will load the dataset into your environment and plot
    # the data.
    
    # Load Training Data
    print('Loading and Visualizing Data ...')
    
    # Load from ex5data1: 
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex5data1.mat")
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    print("\nData keys:")
    print("  ", list(data.keys()))
    
    X = data['X']
    y = data['y']
    
    Xval = data['Xval']
    yval = data['yval']

    Xtest = data['Xtest']

    assert X.ndim == 2
    assert X.shape == (12, 1)
    
    assert y.ndim == 2
    assert y.shape == (12,1)

    y = y.ravel()

    # number of training examples
    m = X.shape[0]
    n = X.shape[1]
    
    # Plot training data
    plt.scatter(X, y, marker='x', color='black');
    plt.xlabel('Change in water level (x)');
    plt.ylabel('Water flowing out of the dam (y)');
    #plt.show()

    # =========== Part 6: Feature Mapping for Polynomial Regression =============
    # One solution to this is to use polynomial regression. You should now
    # complete polyFeatures to map each example into its powers
    
    p = 8
    
    # Map X onto Polynomial Features and Normalize
    X_poly = lr.mapPolyFeatures(X, p)
    X_poly, mu, sigma = lr.featureNormalize(X_poly)  # Normalize
    X_poly = np.insert(X_poly, 0, 1, axis=1) # Add Ones
    
    # Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = lr.mapPolyFeatures(Xtest, p);

    X_poly_test = X_poly_test - mu
    X_poly_test = X_poly_test / sigma
    X_poly_test = np.insert(X_poly_test, 0, 1, axis=1) # Add Ones
    
    # Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = lr.mapPolyFeatures(Xval, p);
    X_poly_val = X_poly_val - mu
    X_poly_val = X_poly_val / sigma
    X_poly_val = np.insert(X_poly_val, 0, 1, axis=1) # Add Ones
    
    print('Normalized Training Example 1:')
    print('  {}  '.format(X_poly[0, :]))
    
    # =========== Part 7: Learning Curve for Polynomial Regression =============
    # Once you have implemented the cost and gradient correctly, the
    # trainLinearReg function will use your cost function to train 
    # regularized linear regression.
    #
    # Write Up Note: The data is non-linear, so this will not give a great 
    #                fit.
    
    # Train linear regression with lambda = 0
    lmda = 0
    theta = trainLinearReg(X_poly, y, lmda)
    
    # Plot fit over the data
    std = np.std(X)
    XX = np.linspace(np.min(X)-std, np.max(X)+std, 100)[:,None]
    X_fit = lr.mapPolyFeatures(XX, p);
    X_fit = X_fit - mu
    X_fit = X_fit / sigma
    X_fit = np.insert(X_fit, 0, 1, axis=1) # Add Ones
    plt.plot(XX, X_fit@theta)
    
    outfile = tmp_path/"ex5data1.png"
    print("Saving figure to {}".format(outfile))
    plt.savefig(outfile)
    
    #plt.show()
    plt.close()

    # =========== Part 5: Learning Curve for Linear Regression =============
    # Next, you should implement the learningCurve function. 
    #
    # Write Up Note: Since the model is underfitting the data, we expect to
    #                see a graph with "high bias" -- slide 8 in ML-advice.pdf 
    
    lmda = 0
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lmda)
    
    ids = [i for i in range(len(error_train))]
    plt.plot(ids, error_train, label='Train')
    plt.plot(ids, error_val, label='Cross Validation')
    plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(lmda))
    plt.ylabel('Error')
    plt.xlabel('Number of training examples')
    plt.legend()
    
    outfile = tmp_path/"ex5data1-learning.png"
    print("Saving figure to {}".format(outfile))
    plt.savefig(outfile)
    
    #plt.show()
    plt.close()

