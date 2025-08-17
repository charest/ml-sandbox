import pytest

import toolbox.file_utils as futils

import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

class BirdImage:
    def loadData(self):
        #  Load an image of a bird
        self.A = plt.imread(TEST_DATA_DIR/'bird_small.png')


@pytest.fixture
def BirdFixture():
    return BirdImage()
