import pytest

import toolbox.file_utils as futils

import numpy as np
import scipy.io as sio

TEST_DATA_DIR = futils.dirname(__file__)

class _MovieDataset():
    def __init__(self):
        data = sio.loadmat(TEST_DATA_DIR/"ex8_movies.mat")
        assert 'Y' in data.keys()
        assert 'R' in data.keys()
        np.testing.assert_array_equal( data["Y"].shape, (1682,943) )
        np.testing.assert_array_equal( data["R"].shape, (1682,943) )
        self.Y = data["Y"]
        self.R = data["R"]

@pytest.fixture
def MovieDataset():
    return _MovieDataset()

class _MovieParams():
    def __init__(self):
        data = sio.loadmat(TEST_DATA_DIR/"ex8_movieParams.mat")
        self.X = data["X"]
        self.Theta = data["Theta"]
        self.num_users = data["num_users"]
        self.num_movies = data["num_movies"]

@pytest.fixture
def MovieParams():
    return _MovieParams()

class _MovieList():
    def __init__(self):
        self.list = []
        with open(TEST_DATA_DIR/"movie_ids.txt", 'r', encoding="ISO-8859-1") as file:
            for line in file:
                split = line.split(' ', 1)
                assert len(split) == 2
                self.list.append( split[1].strip() )
        assert len(self.list) == 1682

@pytest.fixture
def MovieList():
    return _MovieList()
