
import pandas as pd
import numpy as np
from LogReg import LogisticRegression


#Read the data

X_Train = np.genfromtxt('../data/XTrain.csv', delimiter=",")
Y_Train = np.genfromtxt('../data/YTrain.csv',delimiter=",")
Y_Train = Y_Train.reshape((Y_Train.shape[0],1))
X_Test = np.genfromtxt('../data/XTest.csv', delimiter=",")
Y_Test = np.genfromtxt('../data/YTest.csv', delimiter=",")
Y_Test = Y_Test.reshape((Y_Test.shape[0],1))

LogReg=LogisticRegression()

#Uncomment this only after your Logistic Regression Class has been Completed

weight=LogReg.train(X_Train,Y_Train)
Y_predict=np.array(LogReg.predict_label(X_Test,weight))
print (LogReg.calculateAccuracy(Y_predict,Y_Test))

# weight = np.ones((X_Test.shape[1],1))
# LogReg.calculateGradient(weight, X_Test, Y_Test, regLambda=0.01)
# LogReg.update_weight(X_Test, Y_Test, weight)
# LogReg.check_conv(weight, np.zeros((weight.shape)), 15603) #Should be true
# print LogReg.train(X_Test, Y_Test)