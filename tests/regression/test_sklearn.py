
import pytest

from toolbox.modelling import split80_20, randomSplits, leaveOneOut
from toolbox.modelling import specificity, sensitivity, getStats
import toolbox.file_utils as futils

from fixtures.fixture_titanic import TitanicPassenger

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import sklearn.linear_model

TEST_DATA_DIR = futils.dirname(__file__)
TEST_OUTPUT_DIR = futils.dirname(__file__)

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
        print('   ', TitanicPassenger.featureNames[j], '=', model.coef_[0][j])

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
        xVals.append(1.0 - specificity(trueNeg, falsePos))
        yVals.append(sensitivity(truePos, falseNeg))
        p += 0.01
    auroc = sklearn.metrics.auc(xVals, yVals)
    if fname:
        plt.plot(xVals, yVals)
        plt.plot([0,1], [0,1])
        title = title + '\nAUROC = ' + str(round(auroc,3))
        plt.title(title)
        plt.xlabel('1 - specificity')
        plt.ylabel('Sensitivity')
        print("Writing to {}".format(fname))
        plt.savefig(fname)
    return auroc

###############################################################################

def test_sklearn_lr(TitanicExamples):

    rnd.seed(0)
    numSplits = 10
    print('Average of', numSplits, '80/20 splits LR')
    truePos, falsePos, trueNeg, falseNeg = randomSplits(TitanicExamples, lr, numSplits)
    
    assert truePos  == pytest.approx( 61.2, abs=1.e-6)
    assert falsePos == pytest.approx( 21.5, abs=1.e-6)
    assert trueNeg  == pytest.approx( 99.7, abs=1.e-6)
    assert falseNeg == pytest.approx( 26.6, abs=1.e-6)
    
    
    print('Average of LOO testing using LR')
    truePos, falsePos, trueNeg, falseNeg = leaveOneOut(TitanicExamples, lr)
   
    assert truePos  == 301
    assert falsePos == 99
    assert trueNeg  == 520
    assert falseNeg == 126

    #Look at weights
    trainingSet, testSet = split80_20(TitanicExamples)
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
    trainingSet, testSet = split80_20(TitanicExamples)
    model = buildModel(trainingSet)
    print('Try p = 0.1')
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 'Survived', 0.1)
    getStats(truePos, falsePos, trueNeg, falseNeg)
    print('Try p = 0.9')
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 'Survived', 0.9)
    getStats(truePos, falsePos, trueNeg, falseNeg)

###############################################################################

def test_sklearn_roc(TitanicExamples, tmp_path):
    """Receiving operating characteristic"""

    rnd.seed(0)
    trainingSet, testSet = split80_20(TitanicExamples)
    
    auc = buildROC(trainingSet, testSet, 'ROC for Predicting Survival, 1 Split', tmp_path/'roc.png')

    assert auc == pytest.approx(0.860056925996205, abs=1.e-6)

