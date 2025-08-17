import pytest

import toolbox.decomp.pca as pca
import toolbox.file_utils as futils
from  toolbox.covariance.gaussian import *

import numpy as np
import numpy.random as rnd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.optimize import minimize

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################
def unpack(params, num_users, num_movies, num_features):
    split = num_movies * num_features
    X = np.reshape( params[:split], (num_movies, num_features) )
    Theta = np.reshape( params[split:], (num_users, num_features) )
    return X, Theta
 
def pack(X, Theta):
    return np.concatenate( [X.ravel(), Theta.ravel()] )

###############################################################################

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lmda = None):
    """
    COFICOSTFUNC Collaborative filtering cost function
       [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
       num_features, lambda) returns the cost and gradient for the
       collaborative filtering problem.
    """
    
    # Unfold the U and W matrices from params
    X, Theta = unpack(params, num_users, num_movies, num_features)

    pred = X @ Theta.T
    res = pred - Y
    res = R * res
    
    J = 0.5 * np.sum(res * res)
    
    grad_X = res @ Theta
    grad_Theta = res.T @ X

    if lmda != None:
        J += 0.5 * lmda * np.sum( np.square(Theta) )
        J += 0.5 * lmda * np.sum( np.square(X) )
        grad_Theta += lmda * Theta
        grad_X += lmda * X

    grad = pack(grad_X, grad_Theta)
    return J, grad


###############################################################################

def computeNumericalGradient(J, theta, eps=1.e-4):
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
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = eps
        loss1,_ = J(theta - perturb)
        loss2,_ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*eps)
        perturb[p] = 0

    return numgrad

###############################################################################

def normalizeRatings(Y, R):
    """
    NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
    movie (every row)
       [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
       has a rating of 0 on average, and returns the mean rating in Ymean.
    """

    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape);
    for i in range(m):
        idx = np.where(R[i,:] == 1)
        Ymean[i] = np.mean(Y[i,idx])
        Ynorm[i,idx] = Y[i,idx] - Ymean[i]

    return Ynorm, Ymean
    
###############################################################################

def test_cofi_cost_function():    

    # ============== Part 3: Collaborative Filtering Gradient ==============
    # Once your cost function matches up with ours, you should now implement 
    # the collaborative filtering gradient function. Specifically, you should 
    # complete the code in cofiCostFunc.m to return the grad argument.
    print('Checking Gradients (without regularization) ... ')
    
    rnd.seed(0)

    # Create small problem
    X_t = rnd.rand(4, 3)
    Theta_t = rnd.rand(5, 3)
    
    # Zap out most entries
    Y = X_t @ Theta_t.T
    Y[rnd.random_sample(Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1
   
    # Run Gradient Checking
    X = rnd.randn(*X_t.shape);
    Theta = rnd.randn(*Theta_t.shape);
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    def f(t):
        return cofiCostFunc(t, Y, R, num_users, num_movies, num_features)
    numgrad = computeNumericalGradient(f, pack(X, Theta))
   
    cost, grad = f( pack(X, Theta) )
    
    print(np.column_stack((numgrad, grad)))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')
    
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: {}'.format(diff))

    assert diff < 1.e-9

    # ========= Part 4: Collaborative Filtering Cost Regularization ========
    # Now, you should implement regularization for the cost function for 
    # collaborative filtering. You can implement it by adding the cost of
    # regularization to the original cost computation.
    
    lmda = 1.5
    J,_ = cofiCostFunc(pack(X, Theta), Y, R, num_users, num_movies, num_features, lmda)
               
    print('Cost at loaded parameters (lambda = {}): {} '.format(lmda, J))
    print('(this value should be about 31.34)')

###############################################################################

def test_anomoly_recommender(tmp_path, MovieFixture):
    
    # =============== Part 1: Loading movie ratings dataset ================
    # You will start by loading the movie ratings dataset to understand the
    # structure of the data.
    print('Loading movie ratings dataset.')
    MovieFixture.loadRatings()
    
    Y = MovieFixture.Y
    R = MovieFixture.R

    # Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
    # 943 users
    #
    # R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    # rating to movie i
    
    # From the matrix, we can compute statistics like average rating.
    print('Average rating for movie 1 (Toy Story): {} / 5'.format(np.mean(Y[0,R[0,:]])))
    
    # We can "visualize" the ratings matrix by plotting it with imagesc
    plt.imshow(Y, aspect='auto')
    plt.ylabel('Movies')
    plt.xlabel('Users')
    
    outfile = tmp_path / "ex8data2.png"
    print('Writing output to {}'.format(outfile))
    plt.savefig(outfile)

    # ============ Part 2: Collaborative Filtering Cost Function ===========
    # You will now implement the cost function for collaborative filtering.
    # To help you debug your cost function, we have included set of weights
    # that we trained on that. Specifically, you should complete the code in 
    # cofiCostFunc.m to return J.
    
    # Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    MovieFixture.loadWeights()
    
    # Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3
    X = MovieFixture.X[:num_movies,:num_features]
    Theta = MovieFixture.Theta[:num_users, :num_features]
    Y = MovieFixture.Y[:num_movies, :num_users]
    R = MovieFixture.R[:num_movies, :num_users]
    
    # Evaluate cost function
    J,_ = cofiCostFunc(pack(X,Theta), Y, R, num_users, num_movies, num_features, 0)
               
    print('Cost at loaded parameters: {} '.format(J))
    print('(this value should be about 22.22)')

    assert J == pytest.approx(22.224603725685675, abs=1.e-6)

###############################################################################

def test_anomoly_movies(tmp_path, MovieFixture):

    MovieFixture.loadMovies()

    # Initialize my ratings
    movieList = MovieFixture.list
    n = len(movieList)
    my_ratings = np.zeros(n)

    # Check the file movie_idx.txt for id of each movie in our dataset
    # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    my_ratings[0] = 4
    
    # Or suppose did not enjoy Silence of the Lambs (1991), you can set
    my_ratings[97] = 2
    
    # We have selected a few movies we liked / did not like and the ratings we
    # gave are as follows:
    my_ratings[7]   = 3
    my_ratings[12]  = 5
    my_ratings[54]  = 4
    my_ratings[64]  = 5
    my_ratings[66]  = 3
    my_ratings[69]  = 5
    my_ratings[183] = 4
    my_ratings[226] = 5
    my_ratings[355] = 5
    
    print('\nNew user ratings:')
    for i in range(n):
        if my_ratings[i] > 0:
            print('Rated {} for \"{}\"'.format(my_ratings[i], movieList[i]))

    # ================== Part 7: Learning Movie Ratings ====================
    # Now, you will train the collaborative filtering model on a movie rating 
    # dataset of 1682 movies and 943 users
    
    print('Training collaborative filtering...')

    MovieFixture.loadRatings()
    
    # Add our own ratings to the data matrix
    Y = np.column_stack( (my_ratings, MovieFixture.Y) )
    R = np.column_stack( (my_ratings != 0, MovieFixture.R) )
    
    # Normalize Ratings
    Ynorm, Ymean = normalizeRatings(Y, R)
    
    # Useful Values
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10
    
    # Set Initial Parameters (Theta, X)
    rnd.seed(0)
    X = rnd.randn(num_movies, num_features)
    Theta = rnd.randn(num_users, num_features)
    
    initial_params = pack(X, Theta)
    
    # Set options for fmincg
    opts = {
        'maxiter' : 100,
        'disp' : True
    }
    
    # Set Regularization
    lmda = 10
    
    def f(t):
        return cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lmda)
    
    res = minimize(f, initial_params, jac=True, options=opts, method='CG')
    
    # Unfold the returned theta back into U and W
    X, Theta = unpack(res['x'], num_users, num_movies, num_features)
    
    print('Recommender system learning completed.')

    # ================== Part 8: Recommendation for you ====================
    # After training the model, you can now make recommendations by computing
    # the predictions matrix.
    
    p = X @ Theta.T
    my_predictions = p[:,0] + Ymean
    
    
    ix = np.argsort(my_predictions)[::-1]
    print('\nTop recommendations for you:')
    for i in range(10):
        j = ix[i]
        print('Predicting rating {:.1f} for movie \"{}\"'.format(my_predictions[j], movieList[j]))
   
    print('\nOriginal ratings provided:')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated {:.1f} for \"{}\"'.format(my_ratings[i], movieList[i]))
