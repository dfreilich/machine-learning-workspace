import os
import csv
import numpy as np
import NB


# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# TODO: Test logProd function, defined in NB.py

# TODO: Test NB_XGivenY function, defined in NB.py
# xGivenY = NB.NB_XGivenY(XTrain, yTrain, 5, 7)

# TODO: Test NB_YPrior function, defined in NB.py
# prior = NB.NB_YPrior(yTrain)

# TODO: Test NB_Classify function, defined in NB.py
# classify = NB.NB_Classify(xGivenY, prior, XTest)

# TODO: Test classificationError function, defined in NB.py
# error = NB.classificationError(classify, yTest)
# print error

# TODO: Run experiments outlined in HW2 PDF
#Train Classifiers with the first Beta
beta_0 = 5
beta_1 = 7

#Func for Training Classifiers
def trainWith(givenX, givenY, firstBeta, secondBeta):
    xGivenY = NB.NB_XGivenY(givenX, givenY, firstBeta, secondBeta)
    prior = NB.NB_YPrior(givenY)
    return NB.NB_Classify(xGivenY, prior, XTest)

# #First Beta's, Trained with Small
# classifierTrainedWithSmall = trainWith(XTrainSmall, yTrainSmall, beta_0, beta_1)
# trainedWithSmall_Error = NB.classificationError(classifierTrainedWithSmall, yTest)

#First Beta's, Trained with Normal
classifierTrainedWithNormal = trainWith(XTrain, yTrain, beta_0, beta_1)
trainedWithNormal_Error = NB.classificationError(classifierTrainedWithNormal, yTest)

# print 'Trained with Small Error:'
# print trainedWithSmall_Error
print 'Trained with Normal Error:'
print trainedWithNormal_Error

#Training Classifier with second Beta distribution
beta_0=7
beta_1=5

classifierTrainedWithDifferentBeta = trainWith(XTrain, yTrain, beta_0, beta_1)
trainedwithDiffBeta_Error = NB.classificationError(classifierTrainedWithDifferentBeta, yTest)
print 'Trained With Diff Beta'
print trainedwithDiffBeta_Error