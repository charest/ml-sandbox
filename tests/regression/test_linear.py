
import pytest

import toolbox.file_utils as futils
import toolbox.regression.linear as reg

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

TEST_DATA_DIR = futils.dirname(__file__)
    
def printData(X, y):
    assert X.ndim == 2
    n = X.shape[1]
    for i in range(min(10, X.shape[0])):
        print(' x = [ ', end='')
        for j in range(n):
            print('{:6.1f} '.format(X[i,j]), end='')
        print('], y = {:6.1f})'.format(y[i]))

###############################################################################
def test_regression_linear_gradient(tmp_path):

    # ======================= Part 2: Plotting =======================
    X,y = np.loadtxt(TEST_DATA_DIR/"ex1data1.txt", delimiter=",", unpack=True)
    assert X.ndim == 1

    # number of training examples
    m = len(X)
    assert m == 97

    fig, ax = plt.subplots()
    ax.scatter(X,y, marker='x', s=10, color='red', label='Training data')

    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.legend()

    outfile = tmp_path/'ex1data1.png'
    print("\nWriting to {}".format(outfile))
    plt.savefig(outfile)

    # =================== Part 3: Gradient descent ===================
    print('Running Gradient Descent ...')
    
    Xfull = np.c_[ np.ones(m), X ] # Add a column of ones to x
    theta = np.zeros(2) # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # compute and display initial cost
    cost = reg.computeCost(Xfull, y, theta)
    print("Initial cost: J(theta)={:.6e}".format(cost))
    assert cost == pytest.approx(32.072733877455676, abs=1e-6)
    
    # run gradient descent
    theta, _ = reg.gradientDescent(Xfull, y, theta, alpha, iterations)
    
    # print theta to screen
    print('Theta found by gradient descent: {:.6e} {:.6e}'.format(theta[0], theta[1]))
    
    cost = reg.computeCost(Xfull, y, theta)
    print("Final cost: J(theta)={:.6e}".format(cost))
    assert cost == pytest.approx(4.483388256587726, abs=1e-6)
    
    # Plot the linear fit
    yfit = Xfull @ theta
    ax.plot(X, yfit, color='black', label='Linear regression')
    ax.legend()
    #plt.show()   
    
    outfile = tmp_path/'ex1data1-fit.png'
    print("Writing to {}".format(outfile))
    plt.savefig(outfile)
    plt.clf()
 
    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]) @ theta
    print('For population = 35,000, we predict a profit of {:.1f}'.format(predict1*10000))
    predict2 = np.array([1, 7]) @ theta
    print('For population = 70,000, we predict a profit of {:.1f}'.format(predict2*10000))

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')
    
    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, num=100)
    theta1_vals = np.linspace(-1, 4, num=100)
    
    # initialize J_vals to a matrix of 0's
    XX, YY = np.meshgrid(theta0_vals, theta1_vals)
   
    def compute_cost(theta0, theta1):
        t = np.array([theta0, theta1])
        cost = reg.computeCost(Xfull, y, t)
        return cost

    compute_cost_vec = np.vectorize(compute_cost) 
    J_vals = compute_cost_vec(XX, YY)

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, J_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('J_vals')
    
    outfile = tmp_path/'ex1data1-jvals.png'
    print("Writing to {}".format(outfile))
    plt.savefig(outfile)
    #plt.show()

###############################################################################
def test_regression_linear_gradient_multi(tmp_path):

    # ================ Part 1: Feature Normalization ================
    data = np.loadtxt(TEST_DATA_DIR/"ex1data2.txt", delimiter=",")
    assert data.ndim == 2

    # number of training examples
    m = data.shape[0]
    n = data.shape[1]
    assert m == 47
    assert n == 3
    X = data[:, 0:n-1]
    y = data[:, n-1]

    
    # Print out some data points
    print('First 10 examples from the dataset:')
    printData(X, y)

    # Scale features and set them to zero mean
    print('Normalizing Features:')

    Xnorm, mu, sigma = reg.featureNormalize(X)
    printData(Xnorm, y)  
    
    np.testing.assert_allclose(mu, [2000.680851, 3.170213], atol=1.e-6)

    # =================== Part 3: Gradient descent ===================
    print('Running Gradient Descent ...')
    
    Xfull = np.c_[ np.ones(m), Xnorm ] # Add a column of ones to x
    theta = np.zeros(n) # initialize fitting parameters

    # Some gradient descent settings
    iterations = 400
    alpha = 1.0

    # run gradient descent
    theta, J_history = reg.gradientDescent(Xfull, y, theta, alpha, iterations)
    
    # Plot the linear fit
    fig, ax = plt.subplots()
    ax.plot(J_history, color='black')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost J')
    
    outfile = tmp_path/'ex1data2-its.png'
    print("Writing to {}".format(outfile))
    plt.savefig(outfile)
    #plt.show() 
    
    # print theta to screen
    print('Theta found by gradient descent: {}'.format(theta))

    # Estimate the price of a 1650 sq-ft, 3 br house
    Xcheck, _, _ = reg.featureNormalize(np.array([1650, 3]), mu, sigma) 
    price = np.c_[1, Xcheck[np.newaxis,:]] @ theta
    print('Predicted price of a 1650 sq-ft, 3 br house '
          '(using gradient descent): ${:.2f}'.format(price[0]))
    
    assert price[0] == pytest.approx(293081.4643348961, abs=1.e-6)


###############################################################################
def test_regression_linear_normal(tmp_path):

    data = np.loadtxt(TEST_DATA_DIR/"ex1data2.txt", delimiter=",")
    assert data.ndim == 2

    # number of training examples
    m = data.shape[0]
    n = data.shape[1]
    assert m == 47
    assert n == 3
    X = data[:, 0:n-1]
    y = data[:, n-1]

    
    print('Solving with normal equations ...')
    
    Xfull = np.c_[ np.ones(m), X ] # Add a column of ones to x
    
    theta = reg.normalEqn(Xfull, y)
    
    # print theta to screen
    print('Theta found from the normal equations: {}'.format(theta))

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.array([1, 1650, 3]) @ theta
    print('Predicted price of a 1650 sq-ft, 3 br house '
          '(using gradient descent): ${:.2f}'.format(price))
    
    assert price == pytest.approx(293081.4643348961, abs=1.e-6)

