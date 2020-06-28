import math
import numpy as np


# The logProd function takes a vector of numbers in logspace
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
    ## Inputs ##
    # x - 1D numpy ndarray

    ## Outputs ##
    # log_product - float

    log_product = np.sum(x)
    # Uses np.sum to add all the elements in the vector
    return log_product


# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
    ## Inputs ##
    # XTrain - (n by V) numpy ndarray
    # yTrain - 1D numpy ndarray of length n
    # alpha - float
    # beta - float

    ## Outputs ##
    # D - (2 by V) numpy ndarray
    D = np.zeros([2, XTrain.shape[1]])

    # First entry: Sum all XTrain's where yTrain == 0 + beta_0 - 1, and divide by all cases of of xTrain where yTrain= = 0 + beta's - 2
    # Second entry: Sum all xTrain's where yTrain == 1 + beta_0-1 and divide by all cases of xTrain where yTrain==1 + beta's -2
    # numerator = np.sum(XTrain[np.where(yTrain==0)], axis=0) + beta_0 - 1
    # denomenator = XTrain[np.where(yTrain==0)].shape[0]+beta_0+beta_1-2

    def xGivenY(xSet):
        numerator = np.sum(xSet, axis=0) + beta_0 - 1
        denomenator = xSet.shape[0] + beta_0 + beta_1 - 2
        return numerator/denomenator
    D[0] = xGivenY(XTrain[np.where(yTrain==0)])
    D[1] = xGivenY(XTrain[np.where(yTrain==1)])
    return D

# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
    ## Inputs ##
    # yTrain - 1D numpy ndarray of length n

    ## Outputs ##
    # p - float
    numerator = yTrain[np.where(yTrain==0)].shape[0]
    denomenator = yTrain.shape[0] * 1.0
    p = numerator/denomenator
    return p


# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
    ## Inputs ##
    # D - (2 by V) numpy ndarray - The XGivenY array, saying probs of X for a
    # p - float - The prior
    # XTest - (m by V) numpy ndarray

    ## Outputs ##
    # yHat - 1D numpy ndarray of length m
    yHat = np.ones(XTest.shape[0])

    def findProbs(arr, D, Row):
        for i in range(0, Row.shape[0]):
            if(Row[i] == 0):
                arr.append(1-D[i])
            else: #Row[i] is 1
                arr.append(D[i])
        return np.array(arr)

    for i in range(0, XTest.shape[0]):
        # negativeProbPreLog = np.append(findProbs(D[0],XTest[i]),(1-p))
        # positiveProbPreLog = np.append(findProbs(D[1], XTest[i]), p)

        negativeProb = logProd(np.log(findProbs([1-p], D[0], XTest[i])))
        positiveProb = logProd(np.log(findProbs([p], D[1], XTest[i])))

        if positiveProb > negativeProb:
            yHat[i] = 1
        else:
            yHat[i] = 0
    return yHat


# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
    ## Inputs ##
    # yHat - 1D numpy ndarray of length m
    # yTruth - 1D numpy ndarray of length m

    ## Outputs ##
    # error - float

    error = 0
    equalityCheck = np.equal(yHat, yTruth)
    error = equalityCheck[equalityCheck== False].shape[0]/(equalityCheck.shape[0]*1.0)
    return error
