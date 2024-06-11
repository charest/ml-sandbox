import pytest

import toolbox.random_forest as forest
import toolbox.file_utils as futils

import csv
import random

TEST_DATA_DIR = futils.dirname(__file__)

################################################################################

def load_csv(filename):
    """Load a CSV file"""
    file = open(filename, "rt")
    lines = csv.reader(file)
    dataset = list(lines)
    return dataset

################################################################################

def str_column_to_float(dataset, column):
    """Convert string column to float"""
    for row in dataset:
        row[column] = float(row[column].strip())


###############################################################################

def test_forest_gini():
    assert forest.gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]) == 0.5
    assert forest.gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]) == 0.0 

################################################################################

def test_forest_split():
    dataset = [[2.771244718,1.784783929,0],
        [1.728571309,1.169761413,0],
        [3.678319846,2.81281357,0],
        [3.961043357,2.61995032,0],
        [2.999208922,2.209014212,0],
        [7.497545867,3.162953546,1],
        [9.00220326,3.339047188,1],
        [7.444542326,0.476683375,1],
        [10.12493903,3.234550982,1],
        [6.642287351,3.319983761,1]]
    split = forest.get_split(dataset)
    print('\nSplit: [X%d < %.3f]' % ((split['index']+1), split['value']))
    assert split['index'] == 0
    assert split['value'] == 6.642287351

    forest.to_terminal(dataset) == 0

    tree = forest.build_tree(dataset, 3, 1)
    forest.print_tree(tree)

    stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
    for row in dataset:
        prediction = forest.predict(stump, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))
        assert prediction == row[-1]

################################################################################

def test_forest_classify():
    # Test CART on Bank Note dataset
    random.seed(1)
    
    # load and prepare data
    filename = TEST_DATA_DIR/'data_banknote_authentication.txt'
    dataset = load_csv(filename)
    
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    
    # evaluate algorithm
    n_folds = 5
    max_depth = 5
    min_size = 10
    scores = forest.evaluate_algorithm(dataset, forest.decision_tree, n_folds, max_depth, min_size)
    
    for s in scores:
        assert s > 0.96 and s < 1

    print('\nScores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

