import pytest

import toolbox.file_utils as futils

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

def F2C(X):
    assert X.ndim == 2
    n = X.shape[1]
    w = int(np.sqrt(n))

    assert w*w == n

    order = np.reshape( np.arange(n), (w,w) )
    perm = order.T.flatten()
    return X[:,perm]

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

class HandwritingDataset():
    def loadData(self):
        data = sio.loadmat(TEST_DATA_DIR/"handwriting.mat")
        assert 'X' in data.keys()
        assert 'y' in data.keys()

        print("\nData keys:")
        print("  ", list(data.keys()))
        
        self.X = data['X']
        self.y = data['y']

        assert self.X.ndim == 2
        assert self.X.shape == (5000, 400)
        
        assert self.y.ndim == 2
        assert self.y.shape == (5000,1)

        self.y = self.y.ravel()
    
    def loadWeights(self):
        weights = sio.loadmat(TEST_DATA_DIR/"handwriting_weights.mat")

        print("\nWeight keys:")
        print("  ", list(weights.keys()))

        assert 'Theta1'in weights.keys()
        assert 'Theta2'in weights.keys()

        self.Theta1 = weights['Theta1']
        self.Theta2 = weights['Theta2']

        assert self.Theta1.ndim == 2
        assert self.Theta2.ndim == 2
        assert self.Theta1.shape == (25, 401)
        assert self.Theta2.shape == (10, 26)

        print("Shape of Theta1: ", self.Theta1.shape)
        print("Shape of Theta2: ", self.Theta2.shape)

    def F2C(self):
        self.X = F2C( self.X )

    
@pytest.fixture
def HandwritingFixture():
    return HandwritingDataset()

