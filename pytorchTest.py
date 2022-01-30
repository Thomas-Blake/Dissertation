# -*- coding: utf-8 -*-
# This script creates a dataset and compares the Bayes decision rules for normal loss and balanced loss

from cProfile import label
import torch
import math
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from scipy.stats import multivariate_normal
import itertools

from neuralNetTraining import printDecBoundary

class dataGenerator():
    def __init__(self, dist= (0.01,0.01,0.96,0.01,0.01)):
        '''
        dist is a distribution over the distributions defined below
        
        '''
        self.dist = dist
        self.dim=2
        self.distributions = []
        self.mus = []
        self.sigmas = []

        self.mus.append(torch.tensor([3.,4.]))
        self.sigmas.append(torch.tensor([[0.5,1.],[0.,0.5]]))
        self.dist1 = MultivariateNormal(self.mus[0], covariance_matrix=self.sigmas[0])
        self.distributions.append(self.dist1)

        self.mus.append(torch.tensor([-3.,3.]))
        self.sigmas.append(torch.tensor([[0.5,-3],[0.,0.5]]))
        self.dist2 = MultivariateNormal(self.mus[1], covariance_matrix=self.sigmas[1])
        self.distributions.append(self.dist2)

        self.mus.append(torch.tensor([1.,1.]))
        self.sigmas.append(torch.tensor([[10.,2.],[2.,0.5]]))
        self.dist3 = MultivariateNormal(self.mus[2], covariance_matrix=self.sigmas[2])
        self.distributions.append(self.dist3)

        self.mus.append(torch.tensor([-1.,-1.]))
        self.sigmas.append(torch.tensor([[0.5,-1],[0.,0.5]]))
        self.dist4 = MultivariateNormal(self.mus[3], covariance_matrix=self.sigmas[3])
        self.distributions.append(self.dist4)

        self.mus.append(torch.tensor([5.,-1.]))
        self.sigmas.append(torch.tensor([[0.5,0],[0.,0.5]]))
        self.dist4 = MultivariateNormal(self.mus[4], covariance_matrix=self.sigmas[4])
        self.distributions.append(self.dist4)

        self.distCount = len(self.distributions)

    
    def single_sample(self):
        choice = np.random.choice(len(self.dist),p=self.dist)
        return np.append(self.distributions[choice].sample(),choice)

    
    def bulk_sample(self,n):
        data=np.zeros((n,self.dim+1))
        for i in range(n):
            data[i,:]= self.single_sample()
        return data

    def printSample(self,n,ax=None):
        showPlt=False
        if(ax==None):
            showPlt = True
            ax,fig = plt.subplots()
        
        data = self.bulk_sample(n)
        scatter = ax.scatter(data[:,0],data[:,1],marker='.',alpha=0.3, c=data[:,2].astype(int),label=data[:,2].astype(int))
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
        ax.add_artist(legend1)

        if showPlt:
            plt.show()
        else:
            return ax
    
    def generateSample(self,n):
        self.data = self.bulk_sample(n)




class BayesPredictor(dataGenerator):
    def __init__(self):
        super().__init__()
    
    def findContour(self, ax, detail=100,balancedLoss=True):
        x1=np.linspace(-10,10,detail+1)
        x2=np.linspace(-10,10,detail+1)
        xx1,xx2=np.meshgrid(x1,x2)
        z = np.zeros(xx1.shape)
        for i in range(detail+1):
            print(i)
            for j in range(detail+1):
                z[i,j] = self.makePrediction(xx1[i,j],xx2[i,j],balancedLoss)
        
        boundary = {}
        for (i,j) in itertools.combinations(range(self.distCount), 2):
            boundary[(i,j)] = np.array([[0,0]])
        for i in range(len(x1)):
            for j in range(len(x2)-1):
                if z[i,j] != z[i,j+1]:
                    key = (z[i,j],z[i,j+1]) if z[i,j]<z[i,j+1] else (z[i,j+1],z[i,j])
                    boundary[key] = np.append(boundary[key],[[xx1[i,j],xx2[i,j]]],axis=0)
        for j in range(len(x2)):
            for i in range(len(x1)-1):
                if z[i,j] != z[i+1,j]:
                    key = (z[i+1,j],z[i,j]) if z[i+1,j]<z[i,j] else (z[i,j],z[i+1,j])
                    boundary[key] = np.append(boundary[key],[[xx1[i,j],xx2[i,j]]],axis=0)
        noBoundary = []
        for key in boundary.keys():
            if boundary[key].shape == (1,2):
                noBoundary.append(key)
            else:
                boundary[key] = np.delete(boundary[key],0,0)
        for key in noBoundary:
            boundary.pop(key)


        for key in boundary.keys():
            if(balancedLoss):
                ax.scatter(boundary[key][:,0],boundary[key][:,1],c='black',s=0.3)
            else:
                ax.scatter(boundary[key][:,0],boundary[key][:,1],c='red',s=0.3)

        return ax
    
    def makePrediction(self,x1,x2,balancedLoss):
        predict = np.zeros(self.distCount)
        for i in range(self.distCount):
            if balancedLoss:
                predict[i] = multivariate_normal.pdf((x1,x2), mean=self.mus[i], cov=self.sigmas[i])
            else:
                predict[i] = self.dist[i]*multivariate_normal.pdf((x1,x2), mean=self.mus[i], cov=self.sigmas[i])
        return np.argmax(predict)







fig, ax = plt.subplots()

dg = dataGenerator()
ax = dg.printSample(10000,ax)

bp = BayesPredictor()
ax = bp.findContour(ax,1000,True)
ax = bp.findContour(ax,1000,False)
plt.show()

# dg = dataGenerator()
# dg.generateSample(1000)
