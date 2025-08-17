import pytest

import toolbox.file_utils as futils
import toolbox.math_utils as mutils
from toolbox.cluster import Example

import numpy as np
import matplotlib.pyplot as plt

TEST_DATA_DIR = futils.dirname(__file__)

class Patient(Example):
    pass

class CardiacDataset:
    def getData(self, toScale = False):
        """read in data"""
        hrList, stElevList, ageList, prevACSList, classList = [],[],[],[],[]
        cardiacData = open(TEST_DATA_DIR/'cardiacData.txt', 'r')
        for l in cardiacData:
            l = l.split(',')
            hrList.append(int(l[0]))
            stElevList.append(int(l[1]))
            ageList.append(int(l[2]))
            prevACSList.append(int(l[3]))
            classList.append(int(l[4]))
        if toScale:
            hrList = mutils.normalize_by_std(hrList)
            stElevList = mutils.normalize_by_std(stElevList)
            ageList = mutils.normalize_by_std(ageList)
            prevACSList = mutils.normalize_by_std(prevACSList)
        #Build points
        self.points = []
        for i in range(len(hrList)):
            features = np.array([hrList[i], prevACSList[i],\
                                stElevList[i], ageList[i]])
            pIndex = str(i)
            self.points.append(Patient('P'+ pIndex, features, classList[i]))



@pytest.fixture
def CardiacFixture():
    return CardiacDataset()
