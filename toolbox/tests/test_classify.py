
import pytest

import toolbox.cluster as cluster
import toolbox.classify as classify
import toolbox.file_utils as futils

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import sklearn.linear_model


TEST_DATA_DIR = futils.dirname(__file__)

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
        return cluster.minkowskiDist(self.getFeatures(), other.getFeatures(), 2)
                             
    def __str__(self):
        return self.name
                             
###############################################################################

class Passenger(object):
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
        return cluster.minkowskiDist(self.featureVec, other.featureVec, 2)
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
        plt.savefig(fname)

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
        p = Passenger(data['class'][i], data['age'][i],
                      data['gender'][i], data['survived'][i],
                      data['name'][i])
        examples.append(p)
    return examples

###############################################################################

def buildModel(examples):
    featureVecs, labels = [],[]
    for e in examples:
        featureVecs.append(e.getFeatures())
        labels.append(e.getLabel())
    LogisticRegression = sklearn.linear_model.LogisticRegression
    model = LogisticRegression().fit(featureVecs, labels)
    return model
    
###############################################################################
def printModel(model):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # coef_ corresponds to outcome 1
    print('For label', model.classes_[1])
    for j in range(len(model.coef_[0])):
        print('   ', Passenger.featureNames[j], '=', model.coef_[0][j])

###############################################################################

def applyModel(model, testSet, label, prob = 0.5):
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if testSet[i].getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg

###############################################################################

def lr(trainingData, testData, prob = 0.5):
    model = buildModel(trainingData)
    results = applyModel(model, testData, 'Survived', prob)
    return results

###############################################################################

def buildROC(trainingSet, testSet, title, fname=None):
    model = buildModel(trainingSet)
    printModel(model)
    xVals, yVals = [], []
    p = 0.0
    while p <= 1.0:
        truePos, falsePos, trueNeg, falseNeg =\
                               applyModel(model, testSet,
                               'Survived', p)
        xVals.append(1.0 - classify.specificity(trueNeg, falsePos))
        yVals.append(classify.sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = sklearn.metrics.auc(xVals, yVals)
    if fname:
        plt.plot(xVals, yVals)
        plt.plot([0,1], [0,1])
        title = title + '\nAUROC = ' + str(round(auroc,3))
        plt.title(title)
        plt.xlabel('1 - specificity')
        plt.ylabel('Sensitivity')
        plt.savefig(fname)
    return auroc

###############################################################################

def test_classify_num_legs():
   
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

    compareAnimals(animals, 3, 'distances-num-legs.png')

###############################################################################

def test_classify_binary_legs():
   
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

    compareAnimals(animals, 3, 'distances-binary-legs.png')

###############################################################################

@pytest.fixture
def examples():
    examples = buildTitanicExamples(TEST_DATA_DIR / 'TitanicPassengers.txt')
    print('\nFinish processing', len(examples), 'passengers\n')    
    assert len(examples) == 1046
    return examples
    

###############################################################################

def test_classify_knearest(examples):

    knn = lambda training, testSet:classify.KNearestClassify(training, testSet, 'Survived', 3)

    rnd.seed(0)
    numSplits = 10
    print('Average of', numSplits, '80/20 splits using KNN (k=3)')
    truePos, falsePos, trueNeg, falseNeg = classify.randomSplits(examples, knn, numSplits)
        
    assert truePos  == pytest.approx( 58.8, abs=1.e-6)
    assert falsePos == pytest.approx( 19.9, abs=1.e-6)
    assert trueNeg  == pytest.approx(101.3, abs=1.e-6)
    assert falseNeg == pytest.approx( 29.0, abs=1.e-6)
    
    print('Average of LOO testing using KNN (k=3)')
    truePos, falsePos, trueNeg, falseNeg = classify.leaveOneOut(examples, knn)
  
    assert truePos  == 283
    assert falsePos == 98
    assert trueNeg  == 521
    assert falseNeg == 144

###############################################################################

def test_classify_logistic_regression(examples):

    rnd.seed(0)
    numSplits = 10
    print('Average of', numSplits, '80/20 splits LR')
    truePos, falsePos, trueNeg, falseNeg = classify.randomSplits(examples, lr, numSplits)
    
    assert truePos  == pytest.approx( 61.2, abs=1.e-6)
    assert falsePos == pytest.approx( 21.5, abs=1.e-6)
    assert trueNeg  == pytest.approx( 99.7, abs=1.e-6)
    assert falseNeg == pytest.approx( 26.6, abs=1.e-6)
    
    
    print('Average of LOO testing using LR')
    truePos, falsePos, trueNeg, falseNeg = classify.leaveOneOut(examples, lr)
   
    assert truePos  == 301
    assert falsePos == 99
    assert trueNeg  == 520
    assert falseNeg == 126

    #Look at weights
    trainingSet, testSet = classify.split80_20(examples)
    model = buildModel(trainingSet)

    print('model.classes_ =', model.classes_)
    np.testing.assert_array_equal( model.classes_, ['Died', 'Survived'] )
    
    assert model.coef_.shape == (1,5)

    printModel(model)
    
    np.testing.assert_allclose( model.coef_[0], 
        [
            1.139694858103378,
            -0.07239921772789433,
            -1.0672308435428794,
            -0.034512383836116725,
            -2.3332453989603605
        ],
        atol = 1.e-6)
    
    #Look at changing prob
    rnd.seed(0)
    trainingSet, testSet = classify.split80_20(examples)
    model = buildModel(trainingSet)
    print('Try p = 0.1')
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 'Survived', 0.1)
    classify.getStats(truePos, falsePos, trueNeg, falseNeg)
    print('Try p = 0.9')
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 'Survived', 0.9)
    classify.getStats(truePos, falsePos, trueNeg, falseNeg)

###############################################################################

def test_classify_roc(examples):
    """Receiving operating characteristic"""

    rnd.seed(0)
    trainingSet, testSet = classify.split80_20(examples)
    auc = buildROC(trainingSet, testSet, 'ROC for Predicting Survival, 1 Split', 'roc.png')

    assert auc == pytest.approx(0.860056925996205, abs=1.e-6)

