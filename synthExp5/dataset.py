from dis import dis
import torch
from torch import nn
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.distributions.normal import Normal

import matplotlib.pyplot as plt
import itertools
import matplotlib
import matplotlib.patches as mpatches





class CustomSyntheticDataset(Dataset):
  def __init__(self, datasetSize=10000,dist=[0.99,0.01],target_transform=None):
    # Target transform will be one hot encoder
    # We need to change the size of output depending on whether target_transform exists
    self.target_transform = target_transform
    self.datasetSize = datasetSize
    self.dist = dist
    self.dim=1
    self.distributions = []
    self.colors = []
    self.dist1 = Normal(torch.tensor([-2.0]), torch.tensor([1.0]))
    self.distributions.append(self.dist1)
    self.colors.append('blue')
    self.dist2 = Normal(torch.tensor([2.0]), torch.tensor([1.0]))
    self.distributions.append(self.dist2)
    self.colors.append('red')



    self.distCount = len(self.distributions)

    if target_transform == None:
      self.data=torch.zeros(self.datasetSize,self.dim+1)
    else:
      self.data=torch.zeros(self.datasetSize,self.dim+2)

    for i in range(self.datasetSize):
      self.data[i,:]= self.single_sample()


  def __len__(self):
    return self.datasetSize

    
  def single_sample(self):
    choice = np.random.choice(len(self.dist),p=self.dist)
    if self.target_transform:
      return torch.cat((self.distributions[choice].sample(),self.target_transform(choice)),0)
    else:
      return torch.cat((self.distributions[choice].sample(),torch.tensor([choice])),0)

  def __getitem__(self, idx):
    return self.data[idx,0],self.data[idx,1:]
  
  def bulk_sample(self,n):
    if(self.target_transform == None):
      data=torch.zeros(n,self.dim+1)
    else:
      data=torch.zeros(n,self.dim+2)

    for i in range(n):
        data[i,:]= self.single_sample()
    return data

  def printSample(self,ax=None,alpha=0.3):
      showPlt=False
      if(ax==None):
          showPlt = True
          fig, ax = plt.subplots()
      if self.target_transform == None:
        ax.scatter(self.data[:,0],np.zeros(self.datasetSize),marker='.',alpha=alpha, c=self.data[:,1], cmap=matplotlib.colors.ListedColormap(self.colors))
      else:
        ax.scatter(self.data[:,0],np.zeros(self.datasetSize),marker='.',alpha=alpha, c=self.data[:,1:].argmax(1), cmap=matplotlib.colors.ListedColormap(self.colors))

      # if showPlt:
      #     plt.show()
      # else:
      #     return ax
      if showPlt:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.savefig('synthExp5/images/exampleTrainDataset')
      else:
        return ax
  def empiricalWeight(self):
    count = torch.zeros(self.distCount)
    for i in range(self.datasetSize):

      if self.target_transform == None:
        count[int(self.data[i,1].item())] += 1
      else:
        count[int(self.data[i,1:].argmax().item())] += 1
    return count/self.datasetSize
  def count(self):
    count = torch.zeros(self.distCount)
    for i in range(self.datasetSize):

      if self.target_transform == None:
        count[int(self.data[i,1].item())] += 1
      else:
        count[int(self.data[i,1:].argmax().item())] += 1
    return count

  
if __name__ == "__main__":
  #data = CustomSyntheticDataset(dist=expDist(mu=0.9))
  data = CustomSyntheticDataset(dist=[0.7,0.3],datasetSize=100)
  fig, ax = plt.subplots()

  ax = data.printSample(ax)
  plt.savefig('synthExp5/images/test')



