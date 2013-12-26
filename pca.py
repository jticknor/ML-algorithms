#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sys import argv

class pca:
    def __init__(self,X):
        # initialize major parameters
        self.X = X
        
    def plotScores(self,scores):
        # plot PC scores
        plt.plot(scores[:,0],scores[:,1],'ro')
        plt.xlabel('Principal Component #1')
        plt.ylabel('Principal Component #2')
        plt.show()

    def fNormalize(self):
        # find mean value for each column in the data
        self.mu = np.mean(self.X,axis=0)
        m = np.size(self.X,axis=0)
        n = np.size(self.X,axis=1)
        X_new = np.zeros((m,n))
        for i in range(len(self.mu)):
            # subtract mean value for each column to get mean centered
            X_new[:,i] = self.X[:,i]-self.mu[i]
        self.sigma = np.std(X_new,axis=0,ddof=1)
        for i in range(len(self.sigma)):
            # divide by std of column to complete normalization
            X_new[:,i] = X_new[:,i] / self.sigma[i]
        return X_new
    
    def covMatrix(self,Xn):
        m = np.size(Xn,axis=0)
        # compute covariance matrix
        C = (1.0/m)*np.dot(np.transpose(Xn),Xn)
        return C
    
    def computePC(self,Xn):
        cmat =self.covMatrix(Xn)
        # use SVD to obtain pc, eignevectors, eigenvalues
        U, S, V = np.linalg.svd(cmat,full_matrices = 1)
        return U,S,V
        
    def projectPC(self,Xn,K,U):
        m = np.size(Xn,axis=0)
        pcscore = np.zeros((m,K))
        for i in range(m):
            for j in range(K):
                pcscore[i,j] = np.dot(Xn[i,:],U[:,j])
        return pcscore
        
    def varExplained(self,S):
        # compute total variance
        totlatent = sum(S)
        # calculate variance explained by each principal component
        explained = S / totlatent
        return explained


if __name__ == "__main__":
    # Get filename for data (rows are samples, columns are variables)
    filename, datafile = argv
    # Load text file into numpy array for PCA
    X = np.loadtxt(datafile)
    # Create instance of pca
    a = pca(X)
    # Normalize data (mean = 0 , std = 1)
    Xn = a.fNormalize()
    # Compute principal components
    pc,S,V = a.computePC(Xn)
    # Project original (normalized) data to get PCA scores
    scores = a.projectPC(Xn,4,pc)
    # View variance explained by each PC
    explained = a.varExplained(S)
    # Plot first two principal components
    a.plotScores(scores)
