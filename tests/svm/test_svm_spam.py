
import pytest

import toolbox.file_utils as futils
import toolbox.svm as svm

import nltk

import numpy as np
import re

import scipy.io as sio

TEST_DATA_DIR = futils.dirname(__file__)

###############################################################################

def readFile(filename):
    """
    READFILE reads a file and returns its entire contents 
       file_contents = READFILE(filename) reads a file and returns its entire
       contents in file_contents
    """

    f = open(filename, "r")
    lines = f.read()
    return lines

###############################################################################

def getVocabList():
    """
    GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
    cell array of the words
       vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
       and returns a cell array of the words in vocabList.
    """
    
    # Read the fixed vocabulary list
    f = open(TEST_DATA_DIR/"vocab.txt", "r")
    
    # Store all dictionary words in cell array vocab{}
    n = 1899 # Total number of words in the dictionary
    
    # For ease of implementation, we use a struct to map the strings => integers
    # In practice, you'll want to use some form of hashmap
    vocabIdx2Word = []
    vocabWord2Idx = {}
    for j, line in enumerate(f.readlines()):
        l = line.strip().split()
        assert len(l) == 2
        # Word Index (can ignore since it will be = i)
        i = int(l[0])
        assert i-1 == j
        # Actual Word
        vocabIdx2Word.append( l[1] )
        vocabWord2Idx[l[1]] = i-1

    return vocabIdx2Word, vocabWord2Idx

###############################################################################

def processEmail(email_contents, vocabMap):
    """
    PROCESSEMAIL preprocesses a the body of an email and
    returns a list of word_indices 
       word_indices = PROCESSEMAIL(email_contents) preprocesses 
       the body of an email and returns a list of indices of the 
       words contained in the email. 
    """
    
    # Output the email to screen as well
    print('==== Un-Processed Email ====')
    print(email_contents)

    # Init return value
    word_indices = []
    
    # ========================== Preprocess Email ===========================
    
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    
    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);
    
    # Lower case
    email_contents = email_contents.lower()
    
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    
    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    
    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    
    # ========================== Tokenize Email ===========================
    
    # Output the email to screen as well
    print('==== Processed Email ====')
    
    # Process file
    l = 0
    
    # Tokenize and also get rid of any punctuation
    strng = re.split(r"[ @$/#.\-:&*+=\[\]?!(){},\'\">_<;%'\s]", email_contents)
   
    # Remove any non alphanumeric characters
    strng = [re.sub(r"[^a-zA-Z0-9]", '', i) for i in strng]
    
    # Stem the word 
    # (the porterStemmer sometimes has issues, so we use a try catch block)
    ps = nltk.PorterStemmer()
    word_indices = []
    for word in strng:
        new_word = ps.stem(word)
        
        # Skip the word if it is too short
        if len(new_word) == 0:
            continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices 
        #               vector (e.g., word_indices = [word_indices ; 18]; ).
        # 
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        # 
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.
        if word in vocabMap:
            word_indices.append( vocabMap[word] )

        # =============================================================
    
        # Print to screen, ensuring that the output lines are not too long
        if (l + len(word) + 1) > 78:
            print()
            l = 0
        print('{} '.format(word), end='')
        l = l + len(word) + 1
    
    # Print footer
    print('\n=========================')

    return word_indices

###############################################################################

def emailFeatures(word_indices, vocabMap, n):
    """
    EMAILFEATURES takes in a word_indices vector and produces a feature vector
    from the word indices
       x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
       produces a feature vector from the word indices. 
    """
    
    # Total number of words in the dictionary
    #n = 1899
    
    # You need to return the following variables correctly.
    x = np.zeros(n)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return a feature vector for the
    #               given email (word_indices). To help make it easier to 
    #               process the emails, we have have already pre-processed each
    #               email and converted each word in the email into an index in
    #               a fixed dictionary (of 1899 words). The variable
    #               word_indices contains the list of indices of the words
    #               which occur in one email.
    # 
    #               Concretely, if an email has the text:
    #
    #                  The quick brown fox jumped over the lazy dog.
    #
    #               Then, the word_indices vector for this text might look 
    #               like:
    #               
    #                   60  100   33   44   10     53  60  58   5
    #
    #               where, we have mapped each word onto a number, for example:
    #
    #                   the   -- 60
    #                   quick -- 100
    #                   ...
    #
    #              (note: the above numbers are just an example and are not the
    #               actual mappings).
    #
    #              Your task is take one such word_indices vector and construct
    #              a binary feature vector that indicates whether a particular
    #              word occurs in the email. That is, x(i) = 1 when word i
    #              is present in the email. Concretely, if the word 'the' (say,
    #              index 60) appears in the email, then x(60) = 1. The feature
    #              vector should look like:
    #
    #              x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];
   
    
    x[list(set(word_indices))] = 1
    
    # =========================================================================
        
    return x 

###############################################################################

def test_svm_spam(tmp_path):
    
    # ==================== Part 1: Email Preprocessing ====================
    # To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
    # to convert each email into a vector of features. In this part, you will
    # implement the preprocessing steps for each email. You should
    # complete the code in processEmail.m to produce a word indices vector
    # for a given email.
    
    print()
    print('Preprocessing sample email (emailSample1.txt)')
    
    # Load Vocabulary
    vocabList, vocabMap = getVocabList()
    
    
    # Extract Features
    file_contents = readFile(TEST_DATA_DIR/'emailSample1.txt')
    word_indices  = processEmail(file_contents, vocabMap)
    
    # Print Stats
    print('Word Indices: {}'.format(len(word_indices)))

    # ==================== Part 2: Feature Extraction ====================
    # Now, you will convert each email into a vector of features in R^n. 
    # You should complete the code in emailFeatures.m to produce a feature
    # vector for a given email.
    
    print('Extracting features from sample email (emailSample1.txt)');
    
    # Extract Features
    features = emailFeatures(word_indices, vocabMap, len(vocabList))
    
    # Print Stats
    print('Length of feature vector: {}'.format(len(features)))
    print('Number of non-zero entries: {}'.format(sum(features > 0)))

    # =========== Part 3: Train Linear SVM for Spam Classification ========
    # In this section, you will train a linear classifier to determine if an
    # email is Spam or Not-Spam.
    
    # Load the Spam Email dataset
    
    # You will have X, y in your environment
    data = sio.loadmat(TEST_DATA_DIR/"spamTrain.mat")
    assert 'X' in data.keys()
    assert 'y' in data.keys()

    print("spamTrain.mat keys: {}".format(list(data.keys())))
    
    X = data['X']
    y = data['y']
    
    assert X.ndim == 2
    assert X.shape == (4000, 1899)
    
    assert y.ndim == 2
    assert y.shape == (4000,1)

    y = y.ravel()
    
    print('Training Linear SVM (Spam Classification)')
    print('(this may take 1 to 2 minutes) ...')


    
    C = 0.1
    model = svm.train(X, y, C, svm.linearKernel)
    
    p = svm.predict(model, X)
    
    acc = np.mean(p == y)
    print('Training Accuracy: {}'.format(acc * 100))

    # =================== Part 4: Test Spam Classification ================
    # After training the classifier, we can evaluate it on a test set. We have
    # included a test set in spamTest.mat
    
    # Load the test dataset
    # You will have Xtest, ytest in your environment
    data2 = sio.loadmat(TEST_DATA_DIR/"spamTest.mat")
    
    print("spamTest.mat keys: {}".format(list(data2.keys())))
    
    Xtest = data2['Xtest']
    ytest = data2['ytest']
    
    assert Xtest.ndim == 2
    assert Xtest.shape == (1000, 1899)
    
    assert ytest.ndim == 2
    assert ytest.shape == (1000,1)

    ytest = ytest.ravel()
    
    print('Evaluating the trained Linear SVM on a test set ...')
    
    p = svm.predict(model, Xtest)
    
    acc = np.mean(p == ytest)
    print('Test Accuracy: {}'.format(acc * 100))

    # ================= Part 5: Top Predictors of Spam ====================
    # Since the model we are training is a linear SVM, we can inspect the
    # weights learned by the model to understand better how it is determining
    # whether an email is spam or not. The following code finds the words with
    # the highest weights in the classifier. Informally, the classifier
    # 'thinks' that these words are the most likely indicators of spam.
    
    # Sort the weights and obtin the vocabulary list
    idx = np.argsort(model.w)
    idx = idx[::-1]
    
    print('Top predictors of spam: ')
    for i in range(15):
        print(' {:15s} ({:f})'.format(vocabList[idx[i]], model.w[idx[i]]))

    # =================== Part 6: Try Your Own Emails =====================
    # Now that you've trained the spam classifier, you can use it on your own
    # emails! In the starter code, we have included spamSample1.txt,
    # spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
    # The following code reads in one of these emails and then uses your 
    # learned SVM classifier to determine whether the email is Spam or 
    # Not Spam
    
    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!
    filename = TEST_DATA_DIR / 'spamSample1.txt'
    
    # read and predict
    file_contents = readFile(filename)
    word_indices  = processEmail(file_contents, vocabMap)
    x             = emailFeatures(word_indices, vocabMap, len(vocabList))
    p = svm.predict(model, x)
    
    print('processed {}'.format(filename))
    print('spam classification: {}'.format(p))
    print('(1 indicates spam, 0 indicates not spam)')
    
    # Set the file to be read in (change this to spamSample2.txt,
    # emailSample1.txt or emailSample2.txt to see different predictions on
    # different emails types). Try your own emails as well!
    filename = TEST_DATA_DIR / 'spamSample2.txt'
    
    # read and predict
    file_contents = readFile(filename)
    word_indices  = processEmail(file_contents, vocabMap)
    x             = emailFeatures(word_indices, vocabMap, len(vocabList))
    p = svm.predict(model, x)
    
    print('processed {}'.format(filename))
    print('spam classification: {}'.format(p))
    print('(1 indicates spam, 0 indicates not spam)')
