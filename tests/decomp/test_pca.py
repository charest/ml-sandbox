import pytest

import toolbox.decomp.pca as pca
import toolbox.cluster.kmeans as kmeans
import toolbox.file_utils as futils
from  toolbox.decomp.pca import featureNormalize

import numpy as np
import numpy.random as rnd
import scipy.io as sio
import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

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
                      start_j:start_j+example_width] = patch.T / max_val

        curr_ex = curr_ex + 1
        if curr_ex == m:
            break

    # Copy each example into a patch on the display array
    plt.imshow(display_array, cmap='gray', aspect='auto')

    if fname != None:
        print("Writing to {}".format(fname))
        plt.savefig(fname)


###############################################################################

def test_pca(tmp_path):
    
    # ================== Part 1: Load Example Dataset  ===================
    # We start this exercise by using a small dataset that is easily to
    # visualize
    print('\nVisualizing example dataset for PCA.')
    
    # The following command loads the dataset. You should now have the 
    # variable X in your environment
    data = sio.loadmat(TEST_DATA_DIR/"ex7data1.mat")
    assert 'X' in data.keys()

    X = data["X"]
    
    # Visualize the example dataset
    plt.scatter(X[:, 0], X[:, 1])
    
    fname = tmp_path / "ex7data1.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)

    # =============== Part 2: Principal Component Analysis ===============
    # You should now implement PCA, a dimension reduction technique. You
    # should complete the code in pca.m
    print('Running PCA on example dataset.')
    
    # Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = featureNormalize(X)
    
    # Run PCA
    U, S = pca.pca(X_norm)

    # Compute mu, the mean of the each feature
    
    # Draw the eigenvectors centered at mean of data. These lines show the
    # directions of maximum variations in the dataset.
    def drawLine(x1, x2):
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]])

    drawLine(mu, mu+1.5*S[0]*U[:,0])
    drawLine(mu, mu+1.5*S[1]*U[:,1])
    
    fname = tmp_path / "ex7data1-svd.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)
    plt.close()
    
    print('Top eigenvector:');
    print(' U(:,1) = {} {}'.format(U[0,0], U[1,0]))
    print('(you should expect to see -0.707107 -0.707107)')

    np.testing.assert_allclose( U[:2,0], [-0.707107, -0.707107] , atol=1.e-6 )

    # =================== Part 3: Dimension Reduction ===================
    # You should now implement the projection step to map the data onto the 
    # first k eigenvectors. The code will then plot the data in this reduced 
    # dimensional space.  This will show you what the data looks like when 
    # using only the corresponding eigenvectors to reconstruct it.
    print('Dimension reduction on example dataset.')
    
    # Plot the normalized dataset (returned from pca)
    plt.scatter(X_norm[:, 0], X_norm[:, 1], edgecolors='b', facecolors='none')
    
    #  Project the data onto K = 1 dimension
    K = 1
    Z = pca.projectData(X_norm, U, K)
    print('Projection of the first example: {}'.format(Z[0]))
    print('(this value should be about 1.481274)')

    assert Z[0] == pytest.approx(1.481274)
    
    X_rec  = pca.recoverData(Z, U, K)
    print('Approximation of the first example: {} {}'.format(X_rec[0,0], X_rec[0,1]))
    print('(this value should be about  -1.047419 -1.047419)')
    
    np.testing.assert_allclose( X_rec[0,:2], [-1.047419, -1.047419] , atol=1.e-6 )
    
    # Draw lines connecting the projected points to the original points
    plt.scatter(X_rec[:,0], X_rec[:,1], edgecolors='r', facecolors='none')
    for i in range(X_norm.shape[0]):
        plt.plot([X_norm[i,:][0], X_rec[i,:][0]],
             [X_norm[i,:][1], X_rec[i,:][1]],
             linestyle='--', color='k', linewidth=1)
    
    fname = tmp_path / "ex7data1-project.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)
    plt.close()
    
###############################################################################

def test_pca_faces(tmp_path):

    # =============== Part 4: Loading and Visualizing Face Data =============
    # We start the exercise by first loading and visualizing the dataset.
    # The following code will load the dataset into your environment
    print('Loading face dataset.')
    
    # Load Face dataset
    data = sio.loadmat(TEST_DATA_DIR/"ex7faces.mat")
    assert 'X' in data.keys()

    X = data["X"]
    
    # Display the first 100 faces in the dataset
    displayData(X[:100,:], fname=tmp_path/"ex7faces.png")
    plt.close()

    # =========== Part 5: PCA on Face Data: Eigenfaces  ===================
    # Run PCA and visualize the eigenvectors which are in this case eigenfaces
    # We display the first 36 eigenfaces.
    print('Running PCA on face dataset. This mght take a minute or two ...')
    
    # Before running PCA, it is important to first normalize X by subtracting 
    # the mean value from each feature
    X_norm, mu, sigma = pca.featureNormalize(X)
    
    # Run PCA
    U, S = pca.pca(X_norm)
    
    # Visualize the top 36 eigenvectors found
    displayData(U[:, :36].T, fname=tmp_path/"ex7faces-eigen.png")
    plt.close()

    # ============= Part 6: Dimension Reduction for Faces =================
    # Project images to the eigen space using the top k eigenvectors 
    # If you are applying a machine learning algorithm 
    print('Dimension reduction for face dataset.')
    
    K = 100
    Z = pca.projectData(X_norm, U, K)
    
    print('The projected data Z has a size of: {}'.format(Z.shape))


    # ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
    # Project images to the eigen space using the top K eigen vectors and 
    # visualize only using those K dimensions
    # Compare to the original input, which is also displayed
    
    print('Visualizing the projected (reduced dimension) faces.')
    
    X_rec  = pca.recoverData(Z, U, K)
    # Display normalized data
    ax = plt.subplot(1, 2, 1)
    displayData(X_norm[:100,:])
    plt.title('Original faces');
    ax.set_aspect('equal', adjustable='box')
    
    # Display reconstructed data from only k eigenfaces
    ax = plt.subplot(1, 2, 2)
    displayData(X_rec[:100,:])
    plt.title('Recovered faces')
    ax.set_aspect('equal', adjustable='box')
    
    fname = tmp_path/"ex7faces-recon.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)
    
    plt.close()

###############################################################################

def test_pca_bird(tmp_path, BirdImage):

    # === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
    # One useful application of PCA is to use it to visualize high-dimensional
    # data. In the last K-Means exercise you ran K-Means on 3-dimensional 
    # pixel colors of an image. We first visualize this output in 3D, and then
    # apply PCA to obtain a visualization in 2D.
    
    # Re-load the image from the previous exercise and run K-Means on it
    # For this to work, you need to complete the K-Means assignment first
    A = BirdImage

    A = A / 255
    img_size = A.shape
    X = np.reshape(A, (img_size[0] * img_size[1], 3))
    K = 16
    max_iters = 10
    initial_centroids = kmeans.initCentroids(X, K)
    centroids, idx = kmeans.runKMeans(X, initial_centroids, max_iters)
   
    # Sample 1000 random indexes (since working with all the data is
    # too expensive. If you have a fast computer, you may increase this.
    #sel = rnd.choice(X.shape[0], 1000)
    sel = np.arange(X.shape[0])

    colors = np.array([plt.cm.tab20(float(i) / 10) for i in idx[sel]])
    
    # Visualize the data and centroid memberships in 3D
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X[sel, 0], X[sel, 1], X[sel, 2], color=colors)
    plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
    
    fname = tmp_path/"small_bird-centroids.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)

    plt.close()

    # === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
    # Use PCA to project this cloud to 2D for visualization
    
    # Subtract the mean to use PCA
    X_norm, mu, sigma = pca.featureNormalize(X)
    
    # PCA and project the data to 2D
    U, S = pca.pca(X_norm)
    Z = pca.projectData(X_norm, U, 2)
    
    # Plot in 2D
    plt.scatter(Z[sel,0], Z[sel,1], idx[sel], color=colors)
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    
    fname = tmp_path/"small_bird-pca.png"
    print("Writing to {}".format(fname))
    plt.savefig(fname)
