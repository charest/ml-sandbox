
import pytest

import toolbox.file_utils as futils
from toolbox.cluster import minkowskiDist

import numpy as np
import matplotlib.pyplot as plt


TEST_DATA_DIR = futils.dirname(__file__)
TEST_OUTPUT_DIR = futils.dirname(__file__)

###############################################################################

class Animal(object):
    def __init__(self, name, features):
        """Assumes name a string; features a list of numbers"""
        self.name = name
        self.features = np.array(features)
        
    def getName(self):
        return self.name
    
    def getFeatures(self):
        return self.features
    
    def distance(self, other):
        """Assumes other an Animal
           Returns the Euclidean distance between feature vectors
              of self and other"""
        return minkowskiDist(self.getFeatures(), other.getFeatures(), 2)
                             
    def __str__(self):
        return self.name
                             
###############################################################################

def compareAnimals(animals, precision, fname = None):
    """Assumes animals is a list of animals, precision an int >= 0
       Builds a table of Euclidean distance between each animal"""
    #Get labels for columns and rows
    columnLabels = []
    for a in animals:
        columnLabels.append(a.getName())
    rowLabels = columnLabels[:]
    tableVals = []
    #Get distances between pairs of animals
    #For each row
    for a1 in animals:
        row = []
        #For each column
        for a2 in animals:
            if a1 == a2:
                row.append('--')
            else:
                distance = a1.distance(a2)
                row.append(str(round(distance, precision)))
        tableVals.append(row)
    #Produce table
    if fname:
        table = plt.table(rowLabels = rowLabels,
                          colLabels = columnLabels,
                          cellText = tableVals,
                          cellLoc = 'center',
                          loc = 'center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        plt.axis('off')
        print("Writing to {}".format(fname))
        plt.savefig(fname)

###############################################################################

def test_classify_num_legs(tmp_path):
   
    #Actual number of legs
    cobra = Animal('cobra', [1,1,1,1,0])
    rattlesnake = Animal('rattlesnake', [1,1,1,1,0])
    boa = Animal('boa\nconstrictor', [0,1,0,1,0])
    chicken = Animal('chicken', [1,1,0,1,2])
    alligator = Animal('alligator', [1,1,0,1,4])
    dartFrog = Animal('dart frog', [1,0,1,0,4])
    zebra = Animal('zebra', [0,0,0,0,4])
    python = Animal('python', [1,1,0,1,0])
    guppy = Animal('guppy', [0,1,0,0,0])
    animals = [cobra, rattlesnake, boa, chicken, guppy,
               dartFrog, zebra, python, alligator]

    
    compareAnimals(animals, 3, tmp_path/'distances-num-legs.png')

###############################################################################

def test_classify_binary_legs(tmp_path):
   
    #Binary features only           
    cobra = Animal('cobra', [1,1,1,1,0])
    rattlesnake = Animal('rattlesnake', [1,1,1,1,0])
    boa = Animal('boa\nconstrictor', [0,1,0,1,0])
    chicken = Animal('chicken', [1,1,0,1,2])
    alligator = Animal('alligator', [1,1,0,1,1])
    dartFrog = Animal('dart frog', [1,0,1,0,1])
    zebra = Animal('zebra', [0,0,0,0,1])
    python = Animal('python', [1,1,0,1,0])
    guppy = Animal('guppy', [0,1,0,0,0])
    animals = [cobra, rattlesnake, boa, chicken, guppy,
               dartFrog, zebra, python, alligator]

    compareAnimals(animals, 3, tmp_path/'distances-binary-legs.png')

