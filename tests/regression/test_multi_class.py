
import pytest

import toolbox.file_utils as futils
import toolbox.regression.logistic as reg

import numpy as np

import scipy.io as sio
from scipy.optimize import fmin_cg, fmin_ncg, fmin_bfgs
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

TEST_DATA_DIR = futils.dirname(__file__)
    

def F2C(X):
    assert X.ndim == 2
    n = X.shape[1]
    w = int(np.sqrt(n))

    assert w*w == n

    order = np.reshape( np.arange(n), (w,w) )
    perm = order.T.flatten()
    return X[:,perm]

    

###############################################################################
def predictOneVsAll(all_theta, X):
    """
    PREDICT Predict the label for a trained one-vs-all classifier. The labels 
    are in the range 1..K, where K = size(all_theta, 1). 
      p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
      for each example in the matrix X. Note that X contains the examples in
      rows. all_theta is a matrix where the i-th row is a trained logistic
      regression theta vector for the i-th class. You should set p to a vector
      of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
      for 4 examples) 
    """
    assert X.ndim == 2
    m = X.shape[0]
    n = X.shape[1]

    assert all_theta.ndim == 2
    num_labels = all_theta.shape[0]
    
    # Add ones to the X data matrix
    Xfull = np.c_[np.ones(m), X]
    
    # predict
    p = reg.predictOneVsAll(all_theta, Xfull)

    # adjust for 0-indexed arrays
    p += 1

    return p


###############################################################################
def oneVsAll(X, y, num_labels, lmda):
    """
    ONEVSALL trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta 
    corresponds to the classifier for label i
       [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
       logisitc regression classifiers and returns each of these classifiers
       in a matrix all_theta, where the i-th row of all_theta corresponds 
       to the classifier for label i
    """

    # Some useful variables
    assert X.ndim == 2
    m = X.shape[0]
    n = X.shape[1]
    
    assert y.ndim == 1
    assert y.shape[0] == m
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1));
    
    # Add ones to the X data matrix
    Xfull = np.c_[np.ones(m), X]

    # initialize fitting parameters
    initial_theta = np.zeros(n + 1) 

    tfull = np.array([])
    
    # loop over all labels
    for l in range(num_labels):

        # adjust for python 0-index arrays
        ym1 = y - 1
        ybool = ym1 == l
        yint = ybool.astype(int)
        
        print('Training label={} ...'.format(l+1))

        f = lambda t : reg.costFunction(t, Xfull, yint, lmda)
        
        cost0,_ = f(initial_theta)
        
        opts = {
            'maxiter' : 50,
            'disp' : True
        }
        res = minimize(f, initial_theta, jac=True, options=opts, method='CG')
        theta = res['x']
        
        if tfull.size == 0:
            tfull = theta
        else:
            tfull = np.vstack((tfull, theta))

        cost = res['fun']
        print('\t >> cost/cost0: {:.6e}'.format(cost/cost0))


    return tfull
    
        

###############################################################################

def displayData(X, fname=None, width=None):

    # Set example_width automatically if not passed in
    assert X.ndim == 2
    example_width = int(np.sqrt(X.shape[1])) if width == None else width


    # Compute rows, cols
    (m, n) = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    
    # Between images padding
    pad = 1
    
    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width  + pad) ))

    curr_ex = 0
    for i,j in np.ndindex((display_rows, display_cols)):

        if curr_ex == m:
            break 
        # Copy the patch
        patch = np.reshape(X[curr_ex, :], (example_height, example_width))
        
        # Get the max value of the patch
        max_val = max(abs(X[curr_ex, :]))
        start_i = pad + i * (example_height + pad)
        start_j = pad + j * (example_width + pad)
        display_array[start_i:start_i+example_height,
                      start_j:start_j+example_width] = patch / max_val

        curr_ex = curr_ex + 1
        if curr_ex == m:
            break

    # Copy each example into a patch on the display array
    fig, ax = plt.subplots()
    ax.imshow(display_array, cmap='gray', aspect='auto')

    if fname == None:
        plt.show()
    else:
        print("Writing to {}".format(fname))
        plt.savefig(fname)

    plt.close()


###############################################################################

def test_multi_class(tmp_path):

    # Setup the parameters you will use for this part of the exercise
    num_labels = 10          # 10 labels, from 1 to 10   
                             # (note that we have mapped "0" to label 10)

    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # You will be working with a dataset that contains handwritten digits.
    
    data = sio.loadmat(TEST_DATA_DIR/"ex3data1.mat")
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    print("\nData keys:")
    print("  ", list(data.keys()))
    
    X = data['X']
    y = data['y']

    assert X.ndim == 2
    assert X.shape == (5000, 400)
    
    assert y.ndim == 2
    assert y.shape == (5000,1)

    y = y.ravel()

    # number of training examples
    m = X.shape[0]
    n = X.shape[1]

    # repack data
    X = F2C(X)

    # Randomly select 100 data points to display
    np.random.seed(0)
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[0:100], :]

    displayData(sel, tmp_path/'ex3data1.png')

    # ============ Part 2: Vectorize Logistic Regression ============
    #  In this part of the exercise, you will reuse your logistic regression
    #  code from the last exercise. You task here is to make sure that your
    #  regularized logistic regression implementation is vectorized. After
    #  that, you will implement one-vs-all classification for the handwritten
    #  digit dataset.
    #
    
    print('Training One-vs-All Logistic Regression')
    
    lmda = 0.1
    all_theta = oneVsAll(X, y, num_labels, lmda)

    # ================ Part 3: Predict for One-Vs-All ================
    #  After ...
    pred = predictOneVsAll(all_theta, X)
    
    acc = np.mean(pred == y) * 100
    print('Training Set Accuracy: {:f}'.format(acc))

    assert acc == pytest.approx(95.12, abs=1.e-6) 
    
    # Randomly select 2 data points to display
    for i in range(10):
        j = rand_indices[i]
        sel = X[[j], :]
        print('i={}, Predicted {}, Answer {}'.format(j, pred[j], y[j]))
        outfile = tmp_path/'ex3data1_guess{}_expect{}.png'.format(pred[j], y[j])
        displayData(sel, outfile)
