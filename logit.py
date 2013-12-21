#!/usr/bin/python

import numpy as np
import math
from scipy import optimize
from sys import argv


class logit:
    def __init__(self,X,y):
        # initialize major parameters
        self.X = X
        self.y = y

    def sigmoid(self,z):
        # compute the sigmoid value
        m = len(z)
        sig = np.zeros((m,1))
        for i in range(m):
                sig[i] = 1.0/(1.0 + math.exp(-z[i]))
        return sig

    def lrCost(self,theta):
        m = len(self.y)
        h = np.dot(theta,np.transpose(self.X))
        h2 = np.zeros((m,1))
        g = self.sigmoid(h)
        # compute value of the cost function for each data point
        for i in range(m):
            h2[i] = -y[i]*math.log(g[i])-(1.0-y[i])*math.log(1.0-g[i])
        J = (1.0/m)*sum(h2)             
        return J
    
    def lrGrad(self,theta):
        m = len(self.y)
        n = len(theta)
        h = np.dot(theta,np.transpose(self.X))
        g = self.sigmoid(h)
        h3 = np.zeros((m,n))
        grad = np.zeros((n,1))
        for i in range(n):
            for j in range(m):
                h3[j,i] = (g[j]-y[j])
        # compute gradient for each theta value
        for i in range(n):
            grad[i] = (1.0/m)*sum(h3[:,i])
        # flatten gradient array for use in optimization algorithm
        grad = np.ndarray.flatten(grad)
        return grad

    def lrOptimal(self,theta):
        # compute the optimal values for theta using the cost function and gradient specified above
        res1 = optimize.fmin_bfgs(self.lrCost,theta,fprime = self.lrGrad)
        return res1
    
    def lrPredict(self,theta,X2):
        # make predictions for new data values
        m = np.size(X2,axis=0)
        p = np.zeros((m,1))
        prob = self.sigmoid(np.dot(theta,np.transpose(X2)))
        for i in range(m):
            if prob[i] >= 0.5:      # use 0.5 as default threshold. Value can be changed
                p[i] = 1
            else:
                p[i] = 0
        return p
            

if __name__ == "__main__":
    # allow user to specify data file in command line. Program is written for text files
    filename, datafile = argv
    # load text file into numpy array (file delimeter should be ',' as written; if other change next line)   
    data = np.loadtxt(datafile,delimiter=',')
    # Collect input data    
    X = data[:,0:-1]
    # Add column of ones to X for intercept term
    colones = np.ones((np.size(X,axis=0),1))
    X = np.hstack((colones,X))
    # Initialize theta values
    theta = np.zeros((np.size(X,axis=1),1))
    # Collect output data
    y = data[:,-1]
    # Set instance of class logit
    lr = logit(X,y)
    # Build model, optimize theta
    thetavals = lr.lrOptimal(theta)
    # Use model to make predictions
    predictions = lr.lrPredict(thetavals,X)
