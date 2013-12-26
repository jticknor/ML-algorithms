#!/usr/bin/python

import numpy as np
from sys import argv


class kmeans:
    def __init__(self,X,K):
        # initialize major parameters
        self.X = X 
        self.K = K
        self.m = np.size(self.X, axis = 0)
        self.n = np.size(self.X, axis = 1)

    def initialize_Centroids(self):
        # randomly assign centroid locations to input data point locations
        randidx = np.random.permutation(self.m)
        self.C = X[randidx[0:self.K],:]
        
    def find_Closest_Centroids(self):
        # initialize indices and distance between data points
        dist = np.zeros((self.m,self.K))
        idx = np.zeros((self.m,1))

        for i in range(self.m):
            for j in range(self.K):
                # compute distance between data point and centroid
                dist[i,j] = np.linalg.norm(self.X[i,:]-self.C[j,:])**2.0
            # calculate data point index
            idx[i] = np.argmin(dist[i,:])
        return idx

    def compute_Centroids(self,idx):
        # initialize centroids
        centroids = np.zeros((self.K,self.n))
        for i in range(self.K):
            # determine index for all points corresponding to distinct centroid
            vals = np.argwhere(idx == i)[:,0]
            # compute location of centroid
            centroids[i,:] = (1.0/len(vals))*sum(X[vals,:])
        return centroids

    def kmeans_run(self,iters):
        for i in range(iters):
            # calculate centroid index for each data point
            idx = self.find_Closest_Centroids()
            # calculate new location of centroids
            self.C = self.compute_Centroids(idx)
        return idx

if __name__ == "__main__":
    # allow user to specify data file in command line
    filename, datafile = argv
    # load text file into numpy array
    X = np.loadtxt(datafile)
    # number of K initial centroids
    K = 3
    # specify number of iterations for kmeans algorithm
    iters = 10
    # set instance of class kmeans
    output = kmeans(X,K)
    # initialize centroid values
    output.initialize_Centroids()
    # run kmeans algorithm to get clusters
    output.kmeans_run(iters)
