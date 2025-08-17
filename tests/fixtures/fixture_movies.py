import pytest

import toolbox.file_utils as futils

import numpy as np
import scipy.io as sio

TEST_DATA_DIR = futils.dirname(__file__)

class MovieDataset:
    def loadRatings(self):
        data = sio.loadmat(TEST_DATA_DIR/"movies.mat")
        assert 'Y' in data.keys()
        assert 'R' in data.keys()
        np.testing.assert_array_equal( data["Y"].shape, (1682,943) )
        np.testing.assert_array_equal( data["R"].shape, (1682,943) )
        self.Y = data["Y"]
        self.R = data["R"]

    def loadWeights(self):
        data = sio.loadmat(TEST_DATA_DIR/"movie_params.mat")
        self.X = data["X"]
        self.Theta = data["Theta"]
        self.num_users = data["num_users"]
        self.num_movies = data["num_movies"]

    def loadMovies(self):
        self.list = []
        with open(TEST_DATA_DIR/"movie_ids.txt", 'r', encoding="ISO-8859-1") as file:
            for line in file:
                split = line.split(' ', 1)
                assert len(split) == 2
                self.list.append( split[1].strip() )
        assert len(self.list) == 1682

@pytest.fixture
def MovieFixture():
    return MovieDataset()
