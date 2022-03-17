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
  def __init__(self,dist=None,alpha=1,beta=1):
    self.alpha=alpha
    self.beta=beta

    if(dist is None):
        super().__init__()
    else:
        super().__init__(dist=dist)
  
  def findContour(self, ax, detail=100,color='black'):
      # used to be -6,8
      x1=np.linspace(2,6,detail+1)
      x2=np.linspace(-2,2,detail+1)

      
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
    # p1 = P(X|Y_1=y_1)
    for i in range(3):
        p1 = 0
        y1 = i
        for j in range(5):
            p1 += (1/5)*multivariate_normal.pdf((x1,x2), mean=self.mus[5*y1+j], cov=self.sigmas[5*y1+j])
            #print(p1)
        for j in range(5):
            y2 = 5*i+j
            if p1 == 0:
                p2 = 0
            else:
                p2 = multivariate_normal.pdf((x1,x2), mean=self.mus[y2], cov=self.sigmas[y2])/(5*p1)
            predict[y2] = p1*(self.alpha*(p2-1)+self.alpha+self.beta)
    return np.argmax(predict)




    # for i in range(self.distCount):
    #     y1 = i // 5
    #     y2 = i
    #     p1=0
    #     #p1 = sum P(X|Y_2=y_2)P(Y_2=k|Y_1=k)
    #     for j in range(5):
    #         p1 += (1/5)*multivariate_normal.pdf((x1,x2), mean=self.mus[5*y1+j], cov=self.sigmas[5*y1+j])
    #     # p2 = P(Y_2=y_2|Y_1=y_1,X)=P(X|Y_2,Y_1)P(Y_2|Y_1)/P(X|Y_1=y_1)
    #     p2 = multivariate_normal.pdf((x1,x2), mean=self.mus[y2], cov=self.sigmas[2])*(1/5)/p1
    #     #print("p1",p1)
    #     #print("p2",p2)
    #     # predict[i] = p1*(self.alpha*(p2-1)+self.alpha+self.beta)
    #     predict[i] = p1
    #print(predict)
    

    return np.argmax(predict)
  
  def __call__(self,x):
    return torch.tensor(self.makePrediction(x[0][0].detach().numpy(),x[0][1].detach().numpy()),dtype=torch.float,requires_grad=False)



if __name__ == "__main__":

    fig, ax = plt.subplots()

    ds = CustomSyntheticDataset(datasetSize=1000)
    ds.printSample(ax)

    bp = BayesPredictor(alpha=1,beta=1)
    ax, boundary_1 = bp.findContour(ax,400,'black')
    bp = BayesPredictor(alpha=1,beta=1000)
    ax, boundary_1000 = bp.findContour(ax,400,'red')
    #bp.makePrediction(0,0)



    if True:
        with open('./synthExp4/boundaries/boundary_1.pkl', 'wb') as f:
            pickle.dump(boundary_1, f)
        with open('./synthExp4/boundaries/boundary_1000.pkl', 'wb') as f:
            pickle.dump(boundary_1000, f)

    plt.show()




