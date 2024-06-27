import pytest

from toolbox.cluster import minkowskiDist

import toolbox.file_utils as futils

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

class TitanicPassenger(object):
    featureNames = ('C1', 'C2', 'C3', 'age', 'male gender')
    def __init__(self, pClass, age, gender, survived, name):
        self.name = name
        self.featureVec = [0, 0, 0, age, gender]
        self.featureVec[pClass - 1] = 1
        self.label = survived
        self.cabinClass = pClass
    
    ## add constraint C1 + C2 + C3 == 1
    #featureNames = ('C2', 'C3', 'age', 'male gender')
    #def __init__(self, pClass, age, gender, survived, name):
    #    self.name = name
    #    if pClass == 2:
    #        self.featureVec = [1, 0, age, gender]
    #    elif pClass == 3:
    #        self.featureVec = [0, 1, age, gender]
    #    else:
    #        self.featureVec = [0, 0, age, gender]
    #    self.label = survived
    #    self.cabinClass = pClass

    def distance(self, other):
        return minkowskiDist(self.featureVec, other.featureVec, 2)
    def getClass(self):
        return self.cabinClass
    def getAge(self):
        return self.featureVec[3]
    def getGender(self):
        return self.featureVec[4]
    def getName(self):
        return self.name
    def getFeatures(self):
        return self.featureVec[:]
    def getLabel(self):
        return self.label


###############################################################################
        
def getTitanicData(fname):
    data = {}
    data['class'], data['survived'], data['age'] = [], [], []
    data['gender'], data['name'] = [], []
    f = open(fname)
    line = f.readline()
    while line != '':
        split = line.split(',')
        data['class'].append(int(split[0]))
        data['age'].append(float(split[1]))
        if split[2] == 'M':
            data['gender'].append(1)
        else:
            data['gender'].append(0)
        if split[3] == '1':
            data['survived'].append('Survived')
        else:
            data['survived'].append('Died')
        data['name'].append(split[4:])
        line = f.readline()
    return data

###############################################################################
                
def buildTitanicExamples(fileName):
    data = getTitanicData(fileName)
    examples = []
    for i in range(len(data['class'])):
        p = TitanicPassenger(data['class'][i], data['age'][i],
                      data['gender'][i], data['survived'][i],
                      data['name'][i])
        examples.append(p)
    return examples

###############################################################################

@pytest.fixture
def TitanicExamples():
    examples = buildTitanicExamples(TEST_DATA_DIR / 'TitanicPassengers.txt')
    print('\nFinish processing', len(examples), 'passengers\n')    
    assert len(examples) == 1046
    return examples
