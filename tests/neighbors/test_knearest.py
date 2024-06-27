
import pytest

from toolbox.modelling import randomSplits, leaveOneOut
import toolbox.neighbors.knearest as kn
import toolbox.file_utils as futils

import random as rnd


TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

def test_classify_knearest(TitanicExamples):

    knn = lambda training, testSet:kn.KNearestClassify(training, testSet, 'Survived', 3)

    rnd.seed(0)
    numSplits = 10
    print('Average of', numSplits, '80/20 splits using KNN (k=3)')
    truePos, falsePos, trueNeg, falseNeg = randomSplits(TitanicExamples, knn, numSplits)
        
    assert truePos  == pytest.approx( 58.8, abs=1.e-6)
    assert falsePos == pytest.approx( 19.9, abs=1.e-6)
    assert trueNeg  == pytest.approx(101.3, abs=1.e-6)
    assert falseNeg == pytest.approx( 29.0, abs=1.e-6)
    
    print('Average of LOO testing using KNN (k=3)')
    truePos, falsePos, trueNeg, falseNeg = leaveOneOut(TitanicExamples, knn)
  
    assert truePos  == 283
    assert falsePos == 98
    assert trueNeg  == 521
    assert falseNeg == 144

