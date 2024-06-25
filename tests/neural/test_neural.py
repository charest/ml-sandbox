
import pytest

import toolbox.file_utils as futils
import toolbox.regression.logistic as lr
import toolbox.regression.neural as nn

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

TEST_DATA_DIR = futils.dirname(__file__)
    
###############################################################################
def unpack(nn_params, input_layer_size, hidden_layer_size, num_labels):
    split = hidden_layer_size * (input_layer_size + 1)
    Theta1 = np.reshape( nn_params[:split], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape( nn_params[split:], (num_labels, hidden_layer_size + 1))
    return Theta1, Theta2
 
def pack(Theta1, Theta2):
    return np.concatenate( [Theta1.ravel(), Theta2.ravel()] )
    
###############################################################################
def predict(Theta1, Theta2, X):
    
    assert Theta1.ndim == 2
    assert Theta2.ndim == 2
    assert X.ndim == 2

    Xfull = np.insert(X, 0, 1, axis=1)
    z1 = Xfull @ Theta1.T
    a1 = lr.sigmoid(z1)
    
    a2 = np.insert(a1, 0, 1, axis=1)
    z2 = a2 @ Theta2.T
    h = lr.sigmoid(z2)

    p = np.argmax(h, axis=1)

    # +1 for python zero index arrays
    return p+1
    
###############################################################################

def costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmda):

    assert X.ndim == 2
    m = X.shape[0]

    Theta1, Theta2 = unpack(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # Theta1 (25, 401)
    # Theta2 (10, 26)

    # 1. forward propagation

    a1 = np.insert(X, 0, 1, axis=1) # (5000, 401)
    z2 = a1 @ Theta1.T # (5000, 25)
    a2 = lr.sigmoid(z2)   # (5000, 25) 
    
    a2 = np.insert(a2, 0, 1, axis=1) # (5000, 26)
    z3 = a2 @ Theta2.T # (5000, 10)
    h = lr.sigmoid(z3) # (5000, 10)

    cols, col_pos = np.unique(y, return_inverse=True)
    row_pos = np.arange(y.shape[0])

    y_recoded = np.zeros((y.shape[0], num_labels), dtype=y.dtype)
    y_recoded[row_pos, col_pos] = 1 # (5000, 10)

    ans = y_recoded * np.log(h) + (1-y_recoded) * np.log(1 - h)
    J = - np.sum(ans)

    thetaR = Theta1[:,1:]
    J += lmda/2 * np.dot(thetaR.ravel(), thetaR.ravel())
    thetaR = Theta2[:,1:]
    J += lmda/2 * np.dot(thetaR.ravel(), thetaR.ravel())
 
    # 2. Back propagation

    delta3 = h - y_recoded # (5000, 10)
    delta2 = delta3 @ Theta2[:,1:] * lr.sigmoidGradient(z2) # (5000, 10)

    D2 = delta3.T @ a2 # (25, 401)
    D1 = delta2.T @ a1 # (10, 26)

    D1[:,1:] += lmda * Theta1[:,1:]
    D2[:,1:] += lmda * Theta2[:,1:]


    grad = pack( D1/m, D2/m )
    return J/m, grad
    

###############################################################################

def randInitializeWeights(L_in, L_out, epsilon=None):
    """
    RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections
       W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
       of a layer with L_in incoming connections and L_out outgoing 
       connections. 
    
       Note that W should be set to a matrix of size(L_out, 1 + L_in) as
       the column row of W handles the "bias" terms
    """

    if epsilon == None:
        eps = np.sqrt(6)/np.sqrt(L_in+L_out)
    else:
        eps = epsilon

    return np.random.uniform(size=(L_out, 1 + L_in), low=-eps, high=eps)

###############################################################################
def debugInitializeWeights(fan_out, fan_in):
    """
    DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
    incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging
       W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
       of a layer with fan_in incoming connections and fan_out outgoing 
       connections using a fix set of values
    
       Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
       the first row of W handles the "bias" terms
    """
    
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.reshape(np.sin(np.arange(1, fan_out*(1+fan_in)+1)), (fan_out, 1+fan_in)) / 10
    
    return W

###############################################################################

def computeNumericalGradient(J, theta):
    """
    COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    and gives us a numerical estimate of the gradient.
       numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
       gradient of the function J around theta. Calling y = J(theta) should
       return the function value at theta.
    
     Notes: The following code implements numerical gradient checking, and 
            returns the numerical gradient.It sets numgrad(i) to (a numerical 
            approximation of) the partial derivative of J with respect to the 
            i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
            be the (approximately) the partial derivative of J with respect 
            to theta(i).)
    """
    
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for  p in range(theta.size):
        # Set perturbation vector
        perturb[p]= e
        loss1,_ = J(theta - perturb)
        loss2,_ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad

###############################################################################
def checkNNGradients(lmda=0):
    """
    CHECKNNGRADIENTS Creates a small neural network to check the
    backpropagation gradients
       CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
       backpropagation gradients, it will output the analytical gradients
       produced by your backprop code and the numerical gradients (computed
       using computeNumericalGradient). These two gradient computations should
       result in very similar values.
    """
    
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod( np.arange(1, m+1), num_labels)
    
    # Unroll parameters 
    nn_params = pack( Theta1, Theta2 )
    
    # Short hand for cost function
    costFunc = lambda p : costFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmda)
    
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar. 
    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')
    
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    
    print('If your backpropagation implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: {}'.format(diff))
    assert diff < 1.e-9

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

def test_neural_feedforward_predict(tmp_path):

    # Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   
                             # (note that we have mapped "0" to label 10)
    
    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # You will be working with a dataset that contains handwritten digits.
    
    # Load Training Data
    print('\nLoading and Visualizing Data ...')
    
    
    data = sio.loadmat(TEST_DATA_DIR/"ex4data1.mat")
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

    # ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized 
    # neural network parameters.
    
    print('Loading Saved Neural Network Parameters ...')
    
    # Load the weights into variables Theta1 and Theta2
    weights = sio.loadmat(TEST_DATA_DIR/"ex4weights.mat")
    
    print("\nWeight keys:")
    print("  ", list(weights.keys()))

    assert 'Theta1'in weights.keys()
    assert 'Theta2'in weights.keys()

    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    assert Theta1.ndim == 2
    assert Theta2.ndim == 2
    assert Theta1.shape == (25, 401)
    assert Theta2.shape == (10, 26)

    print("Shape of Theta1: ", Theta1.shape)
    print("Shape of Theta2: ", Theta2.shape)

    # ================= Part 3: Implement Predict =================
    # After training the neural network, we would like to use it to predict
    # the labels. You will now implement the "predict" function to use the
    # neural network to predict the labels of the training set. This lets
    # you compute the training set accuracy.
    
    pred = predict(Theta1, Theta2, X)

    acc = np.mean(pred == y) * 100
    print('Training Set Accuracy: {:f}'.format(acc))

    assert acc == pytest.approx(97.52, abs=1.e-6)
    

###############################################################################

def test_neural_feedforward(tmp_path):

    # Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 1 to 10   
                             # (note that we have mapped "0" to label 10)
    
    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset. 
    # You will be working with a dataset that contains handwritten digits.
    
    # Load Training Data
    print('\nLoading and Visualizing Data ...')
    
    
    data = sio.loadmat(TEST_DATA_DIR/"ex4data1.mat")
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

    # ================ Part 2: Loading Pameters ================
    # In this part of the exercise, we load some pre-initialized 
    # neural network parameters.
    
    print('Loading Saved Neural Network Parameters ...')
    
    # Load the weights into variables Theta1 and Theta2
    weights = sio.loadmat(TEST_DATA_DIR/"ex4weights.mat")
    
    print("\nWeight keys:")
    print("  ", list(weights.keys()))

    assert 'Theta1'in weights.keys()
    assert 'Theta2'in weights.keys()

    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    assert Theta1.ndim == 2
    assert Theta2.ndim == 2
    assert Theta1.shape == (25, 401)
    assert Theta2.shape == (10, 26)

    print("Shape of Theta1: ", Theta1.shape)
    print("Shape of Theta2: ", Theta2.shape)

    # Unroll parameters 
    nn_params = pack( Theta1, Theta2 )
    
    print("Shape of nn_params: ", nn_params.shape)


    # ================ Part 3: Compute Cost (Feedforward) ================
    # To the neural network, you should first start by implementing the
    # feedforward part of the neural network that returns the cost only. You
    # should complete the code in nnCostFunction.m to return cost. After
    # implementing the feedforward to compute the cost, you can verify that
    # your implementation is correct by verifying that you get the same cost
    # as us for the fixed debugging parameters.
    #
    # We suggest implementing the feedforward cost *without* regularization
    # first so that it will be easier for you to debug. Later, in part 4, you
    # will get to implement the regularized cost.
    #
    print('Feedforward Using Neural Network ...')
    
    # Weight regularization parameter (we set this to 0 here).
    lmda = 0
    
    J,_ = costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmda)
    
    print('Cost at parameters (loaded from ex4weights): {}'.format(J))
    print('(this value should be about 0.287629)')

    assert J == pytest.approx(0.287629, abs=1.e-6)

    # =============== Part 4: Implement Regularization ===============
    # Once your cost function implementation is correct, you should now
    # continue to implement the regularization with the cost.
    
    print('Checking Cost Function (w/ Regularization) ... ')
    
    # Weight regularization parameter (we set this to 1 here).
    lmda = 1
    
    J,_ = costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmda)
    
    print('Cost at parameters (loaded from ex4weights): {} '.format(J))
    print('(this value should be about 0.383770)')
    
    assert J == pytest.approx(0.383770, abs=1.e-6)

    # ================ Part 5: Sigmoid Gradient  ================
    # Before you start implementing the neural network, you will first
    # implement the gradient for the sigmoid function. You should complete the
    # code in the sigmoidGradient.m file.
    
    print('Evaluating sigmoid gradient...')
    
    xtest = np.array([1, -0.5, 0, 0.5, 1])
    g = lr.sigmoidGradient(xtest)
    print('Sigmoid gradient evaluated at {}:', xtest)
    print('  ', g)

    # ================ Part 6: Initializing Pameters ================
    # In this part of the exercise, you will be starting to implment a two
    # layer neural network that classifies digits. You will start by
    # implementing a function to initialize the weights of the neural network
    # (randInitializeWeights.m)
    
    print('Initializing Neural Network Parameters ...')
    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size, 0.12)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels, 0.12)
    
    # Unroll parameters
    initial_nn_params = pack( initial_Theta1, initial_Theta2 )

    # =============== Part 7: Implement Backpropagation ===============
    # Once your cost matches up with ours, you should proceed to implement the
    # backpropagation algorithm for the neural network. You should add to the
    # code you've written in nnCostFunction.m to return the partial
    # derivatives of the parameters.
    print('Checking Backpropagation... ');
    
    # Check gradients by running checkNNGradients
    checkNNGradients()

    # =============== Part 8: Implement Regularization ===============
    # Once your backpropagation implementation is correct, you should now
    # continue to implement the regularization with the cost and gradient.
    
    print('Checking Backpropagation (w/ Regularization) ... ')
    
    # Check gradients by running checkNNGradients
    lmda = 3
    checkNNGradients(lmda)
    
    # Also output the costFunction debugging values
    debug_J,_  = costFunction(nn_params, input_layer_size,
                              hidden_layer_size, num_labels, X, y, lmda)
    
    print('Cost at (fixed) debugging parameters (w/ lambda = {}): {} '.format(lmda, debug_J))
    print('(this value should be about 0.576051)'.format(debug_J))
    
    assert debug_J == pytest.approx(0.576051, abs=1.e-6)

    # =================== Part 8: Training NN ===================
    # You have now implemented all the code necessary to train a neural 
    # network. To train your neural network, we will now use "fmincg", which
    # is a function which works similarly to "fminunc". Recall that these
    # advanced optimizers are able to train our cost functions efficiently as
    # long as we provide them with the gradient computations.
    print('\nTraining Neural Network... ')
    
    # After you have completed the assignment, change the MaxIter to a larger
    # value to see how more training helps.
    opts = {
        'maxiter' : 50,
        'disp' : True
    }
    
    # You should also try different values of lambda
    lmbd = 1
    
    # Create "short hand" for the cost function to be minimized
    def f_and_jac(p):
        return costFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmda)
    
    # Now, costFunction is a function that takes in only one argument (the
    # neural network parameters)
    nn_res = minimize(f_and_jac, initial_nn_params, jac=True, options=opts, method="CG")
    nn_params = nn_res['x']
    
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1, Theta2 = unpack(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # ================= Part 9: Visualize Weights =================
    # You can now "visualize" what the neural network is learning by 
    # displaying the hidden units to see what features they are capturing in 
    # the data.
    
    print('Visualizing Neural Network... ')
    
    displayData(Theta1[:, 1:], tmp_path/'ex4data1-weights.png')

    # ================= Part 10: Implement Predict =================
    # After training the neural network, we would like to use it to predict
    # the labels. You will now implement the "predict" function to use the
    # neural network to predict the labels of the training set. This lets
    # you compute the training set accuracy.
    
    pred = predict(Theta1, Theta2, X)
    
    acc = np.mean(pred == y) * 100
    print('Training Set Accuracy: {}'.format(acc))

    assert acc > 95
