# Try to set up a new distribution with one large set and many smaller classes dotted around the larger one
# A more complex example would be one large class surrounded by many medium classses surrounded by many smaller distributions

# -*- coding: utf-8 -*-
# This script creates a dataset and compares the Bayes decision rules for normal loss and balanced loss

from cProfile import label
from types import ModuleType
import torch
import math
import matplotlib.pyplot as plt
import matplotlib
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from scipy.stats import multivariate_normal
import itertools
import pickle
from dataset import CustomSyntheticDataset,distCreater
# from neuralNetTraining import printDecBoundary


class BayesPredictor(CustomSyntheticDataset):
  def __init__(self,balancedLoss,dist=None):
    self.balancedLoss = balancedLoss

    if(dist is None):
        super().__init__()
    else:
        super().__init__(dist=dist)
  
  def findContour(self, ax, detail=100,color='black'):
      x1=np.linspace(-7,22,detail+1)
      x2=np.linspace(-20,7,detail+1)

      
      xx1,xx2=np.meshgrid(x1,x2)
      z = np.zeros(xx1.shape)
      for i in range(detail+1):
          if(i % 10 == 0):
            print(i)
          for j in range(detail+1):
              z[i,j] = self.makePrediction(xx1[i,j],xx2[i,j])
      
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
        ax.scatter(boundary[key][:,0],boundary[key][:,1],c=color,s=0.3)

      return ax , boundary
  
  def makePrediction(self,x1,x2):
    predict = np.zeros(self.distCount)
    for i in range(self.distCount):
      if self.balancedLoss:
          predict[i] = multivariate_normal.pdf((x1,x2), mean=self.mus[i], cov=self.sigmas[i])
      else:
          predict[i] = self.dist[i]*multivariate_normal.pdf((x1,x2), mean=self.mus[i], cov=self.sigmas[i])
    return np.argmax(predict)
  
  def __call__(self,x):
    return torch.tensor(self.makePrediction(x[0][0].detach().numpy(),x[0][1].detach().numpy()),dtype=torch.float,requires_grad=False)



if __name__ == "__main__":

    fig, ax = plt.subplots()
    dist = np.load('synthExp3/dist.npy')

    #ds = CustomSyntheticDataset(datasetSize=100000)
    #dg.printSample(10000,ax)

    bp_normalBayes = BayesPredictor(False,dist)
    #vfunc = np.vectorize(bp_normalBayes.makePrediction)
    ax, boundary_normal = bp_normalBayes.findContour(ax,100,'black')
    #ax, boundary_normal = printDecBoundary(ax, vfunc,detail=200,modeltype="numpy",distCount=33,a=-20,b=20)

    if False:
        with open('./synthExp3/bayesNormalBoundary.pkl', 'wb') as f:
            pickle.dump(boundary_normal, f)

    bp_balancedBayes = BayesPredictor(True,dist)
    ax, boundary_balanced = bp_balancedBayes.findContour(ax,100,'blue')
    #vfunc = np.vectorize(bp_balancedLoss.makePrediction)


    # ax, boundary_balanced = printDecBoundary(ax, vfunc, detail=200, modeltype="numpy",distCount=33,a=-20,b=20)

    if False:
        with open('./synthExp3/bayesBalancedBoundary.pkl', 'wb') as f:
            pickle.dump(boundary_balanced, f)

    plt.show()




