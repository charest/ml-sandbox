import pytest

import toolbox.file_utils as futils

import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

@pytest.fixture
def BirdImage():
    #  Load an image of a bird
    A = plt.imread(TEST_DATA_DIR/'bird_small.png')
    return A
