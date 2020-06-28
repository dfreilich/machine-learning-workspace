#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    

    
    
    def calculateGradient(self, weight, X, Y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
        
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is (d+1)-by-1 dimensional numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''
        #Takes in the weight matrix
        def lrHypothesis_Theta(Theta, xRow):
            # To use np.dot, make sure that you are multiplying matrices - and for that, make sure that xRow is a matrix, and not an array
            #may have to try np.dot(Theta, xRow)
            return self.sigmoid(np.dot(np.transpose(Theta), xRow))

        # Make Array filled with LogReg Hypothesis function, so we don't compute it every time
        weight = np.array(weight)
        hyp = np.zeros((X.shape[0],1))
        for i in range(0, hyp.shape[0]):
            hyp[i] = lrHypothesis_Theta(weight, X[i])

        Gradient = np.zeros((weight.shape[0],1))
        for j in range(0, Gradient.shape[0]):
            reg = regLambda*weight[j]
            subt = np.subtract(hyp, Y)
            mult = np.transpose(subt)[0]*X[:,j]
            Gradient[j] = np.sum(mult, axis=0) + reg
        Gradient[0] = Gradient[0] - (regLambda*weight[0])

        return Gradient

    def sigmoid(self, Z):
        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
       
        '''

        denominator = 1+np.exp(-Z)
        sigmoid = 1/denominator
        
        return sigmoid

    def update_weight(self,X,Y,weight):
        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        new_weight = weight - (self.alpha * self.calculateGradient(weight, X, Y, self.regLambda))
        
        return new_weight
    
    def check_conv(self,weight,new_weight,epsilon):
        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged (diff below given threshold epsilon), otherwise False

        '''
        def l2Norm(Theta):
            return np.sum(np.sqrt(Theta**2))

        if (l2Norm(new_weight-weight) <= epsilon):
            return True
        else:
            return False
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape
        
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1))

        num_iterations = 0
        self.new_weight = self.update_weight(X, Y, self.weight)
        while ((self.check_conv(self.weight, self.new_weight, self.epsilon) == False) and num_iterations < self.maxNumIters):
            self.weight = self.new_weight
            self.new_weight = self.update_weight(X, Y, self.weight)
            num_iterations += 1

        return self.weight

    def predict_label(self, X, weight):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]

        # print weight
        def lrHypothesis_Theta(Theta, xRow):
            # To use np.dot, make sure that you are multiplying matrices - and for that, make sure that xRow is a matrix, and not an array
            #may have to try np.dot(Theta, xRow)
            return self.sigmoid(np.dot(np.transpose(Theta), xRow))

        hyp = np.zeros((X.shape[0],1))
        for i in range(0, hyp.shape[0]):
            hyp[i] = lrHypothesis_Theta(weight, X[i])

        result = np.zeros((X.shape[0],1))
        for i in range(0, hyp.shape[0]):
            if(hyp[i]) > 0.5:
                result[i] = 1
            else:
                result[i] = 0

        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''

        equalityCheck = np.equal(Y_predict, Y_test)
        Accuracy = (equalityCheck[equalityCheck == True].shape[0] / (equalityCheck.shape[0] * 1.0)) * 100

        return Accuracy
    
        