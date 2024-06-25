
import pytest

import toolbox.file_utils as futils
import toolbox.regression.logistic as lr

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

def test_regression_sigmoid():
    assert lr.sigmoid(0) == pytest.approx(0.5, abs=1.e-6)
    assert lr.sigmoid(1e6) == pytest.approx(1, abs=1.e-6)
    assert lr.sigmoid(-1e6) == pytest.approx(0, abs=1.e-6)

###############################################################################

def test_regression_logistic(tmp_path):
    
    data = np.loadtxt(TEST_DATA_DIR/"ex2data1.txt", delimiter=",")
    assert data.ndim == 2

    # number of training examples
    m = data.shape[0]
    n = data.shape[1]
    assert m == 100
    assert n == 3
    X = data[:, 0:n-1]
    y = data[:, n-1]
    
    # ==================== Part 1: Plotting ====================
    print('\nPlotting data with + indicating (y = 1) examples and o '
          'indicating (y = 0) examples.')
    

    pos=np.where(y==1)
    neg=np.where(y==0)

    fig, ax = plt.subplots()
    ax.scatter(X[pos,0], X[pos,1], marker='+', color='black', label='Admitted')
    ax.scatter(X[neg,0], X[neg,1], marker='o', color='red', label='Not admitted')

    plt.ylabel('Exam 1 score')
    plt.xlabel('Exam 2 score')
    ax.legend()

    outfile = tmp_path/'ex2data1.png'
    print("\nWriting to {}".format(outfile))
    plt.savefig(outfile)
    #plt.show()
    
    # ============ Part 2: Compute Cost and Gradient ============

    # Initialize fitting parameters
    Xfull = np.c_[ np.ones(m), X ] # Add a column of ones to x
    initial_theta = np.zeros(n) # initialize fitting parameters
    
    # Compute and display initial cost and gradient
    cost, grad = lr.costFunction(initial_theta, Xfull, y)
    
    print('Cost at initial theta (zeros): ', cost)
    print('Gradient at initial theta (zeros): ', grad)

    assert cost == 0.6931471805599453
    
    # ============= Part 3: Optimizing using fminunc  =============
    #  In this exercise, you will use a built-in function (fminunc) to find the
    #  optimal parameters theta.
    
    #  Set options for fminunc
    #options = lr.optimset('GradObj', 'on', 'MaxIter', 400);

    f = lambda t: lr.costFunction(t, Xfull, y)   

    opts = {
        'maxiter' : 400,
        'disp' : True
    }
    res = minimize(f, initial_theta, jac=True, options=opts, method='bfgs')
    cost = res['fun']
    theta = res['x']
    
    # Print theta to screen
    print('Cost at theta found by fminunc: ', cost);
    print('theta:');
    print('  ', theta);

    assert cost == pytest.approx(0.20349770158944375)
    
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.array([np.min(X[:,1])-2., np.max(X[:,1])+2.])

    # Calculate the decision boundary line
    plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
    ax.plot(plot_x, plot_y, color='blue', label='Decision boundary')
    ax.legend()
    
    outfile = tmp_path/'ex2data1-bnd.png'
    print("\nWriting to {}".format(outfile))
    plt.savefig(outfile)
    #plt.show()
    
    # ============== Part 4: Predict and Accuracies ==============
    
    #  Predict probability for a student with score 45 on exam 1 
    #  and score 85 on exam 2 
    
    Xcheck = [1, 45, 85] 
    prob = lr.sigmoid(Xcheck @ theta)
    print('For a student with scores 45 and 85, we predict an admission '
          'probability of {}\n'.format(prob))
    assert prob == pytest.approx(0.7762907240588944, abs=1.e-6)
    
    #% Compute accuracy on our training set
    p = lr.predict(theta, Xcheck)
    acc = np.mean(p == y) 
    
    print('Train Accuracy: {}'.format( acc * 100) )
    
    assert acc == pytest.approx(0.6, abs=1.e-6)

###############################################################################

def test_regression_logistic_reg(tmp_path):
    
    data = np.loadtxt(TEST_DATA_DIR/"ex2data2.txt", delimiter=",")
    assert data.ndim == 2

    # number of training examples
    m = data.shape[0]
    n = data.shape[1]
    assert m == 118
    assert n == 3
    X = data[:, 0:n-1]
    y = data[:, n-1]
    
    # ==================== Part 1: Plotting ====================
    print('\nPlotting data with + indicating (y = 1) examples and o '
          'indicating (y = 0) examples.')
    

    pos=np.where(y==1)
    neg=np.where(y==0)

    fig, ax = plt.subplots()
    ax.scatter(X[pos,0], X[pos,1], marker='+', color='black', label='y = 1')
    ax.scatter(X[neg,0], X[neg,1], marker='o', color='red', label='y = 0')

    plt.ylabel('Microchip Test 1')
    plt.xlabel('Microchip Test 2')
    ax.legend()

    outfile = tmp_path/'ex2data2.png'
    print("\nWriting to {}".format(outfile))
    plt.savefig(outfile)
    #plt.show()


    # =========== Part 1: Regularized Logistic Regression ============
    # In this part, you are given a dataset with data points that are not
    # linearly separable. However, you would still like to use logistic 
    # regression to classify the data points. 
    #
    # To do so, you introduce more features to use -- in particular, you add
    # polynomial features to our data matrix (similar to polynomial
    # regression).
    
    # Add Polynomial Features
    
    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = lr.mapPolyFeature(X[:,0], X[:,1])

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])
    
    # Set regularization parameter lambda to 1
    lmda = 1
    
    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = lr.costFunction(initial_theta, X, y, lmda)
    
    print('Cost at initial theta (zeros): {}'.format(cost))

    # ============= Part 2: Regularization and Accuracies =============
    # Optional Exercise:
    # In this part, you will get to try different values of lambda and 
    # see how regularization affects the decision coundart
    #
    # Try the following values of lambda (0, 1, 10, 100).
    #
    # How does the decision boundary change when you vary lambda? How does
    # the training set accuracy vary?
    
    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])
    
    # Set regularization parameter lambda to 1 (you should vary this)
    lmda = 1
    
    # Set Options
    opts = { 
        'maxiter': 400,
        'disp': True
    }
    
    f = lambda t : lr.costFunction(t, X, y, lmda)

    # Optimize
    res = minimize(f, initial_theta, jac=True, options=opts, method='bfgs')
    theta = res['x']
    
    # Plot Boundary
   
    # Here is the grid range
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    XX, YY = np.meshgrid(u, v)
    # Evaluate z = theta*x over the grid
    z = lr.mapPolyFeature(XX.ravel(), YY.ravel()) @ theta
    z = z.reshape(XX.shape)

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    plt.contour(u, v, z, levels=[0], label='Decision boundary')
    plt.title('lambda = {}'.format(lmda))
    
    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    
    plt.legend()
    
    outfile = tmp_path/'ex2data2-boundary.png'
    print("\nWriting to {}".format(outfile))
    plt.savefig(outfile)
    #plt.show()
    
    # Compute accuracy on our training set
    p = lr.predict(theta, X)
    
    acc = np.mean(p == y) * 100
    print('Train Accuracy: {}'.format(acc))

    assert acc == pytest.approx(83.05084745762711, abs=1e-6)
