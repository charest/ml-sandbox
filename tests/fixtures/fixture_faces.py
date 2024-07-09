import pytest

import toolbox.file_utils as futils

import numpy as np
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

@pytest.fixture
def FacesDataset():
    data = sio.loadmat(TEST_DATA_DIR/"ex7faces.mat")
    assert 'X' in data.keys()
    return data["X"]
