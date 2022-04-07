import torch
from torch import nn
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import itertools
import matplotlib


class CustomSyntheticDataset(Dataset):
  def __init__(self, datasetSize=100000,dist=(0.9,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01),target_transform=None):
    # Target transform will be one hot encoder
    # We need to change the size of output depending on whether target_transform exists
    self.target_transform = target_transform
    self.datasetSize = datasetSize
    self.dist = dist
    self.dim=2
    self.distributions = []
    self.mus = []
    self.sigmas = []
    self.colors = []

    self.mus.append(torch.tensor([0.,0.]))
    #self.sigmas.append(torch.tensor([[2.,0.],[0.,2.]]))
    self.sigmas.append(torch.tensor([[4.,0.],[0.,4.]]))
    self.dist1 = MultivariateNormal(self.mus[0], covariance_matrix=self.sigmas[0])
    self.distributions.append(self.dist1)
    self.colors.append('m')

    for i in range(10):
      self.mus.append(torch.tensor([5*np.cos(2*np.pi*i/10),5*np.sin(2*np.pi*i/10)],dtype=torch.float))
      self.sigmas.append(torch.tensor([[0.1,0.],[0.,0.1]]))
      self.distributions.append(MultivariateNormal(self.mus[i+1], covariance_matrix=self.sigmas[i+1]))
      if(i % 2 == 0):
        self.colors.append('b')
      else:
        self.colors.append('r')

    self.distCount = len(self.distributions)

    self.data=np.zeros((self.datasetSize,self.dim+5))
    if target_transform == None:
      self.data=torch.zeros(self.datasetSize,self.dim+1)
    else:
      self.data=torch.zeros(self.datasetSize,self.dim+11)

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
    return self.data[idx,0:2],self.data[idx,2:]
  
  def bulk_sample(self,n):
    if(self.target_transform == None):
      data=torch.zeros(n,self.dim+1)
    else:
      data=torch.zeros(n,self.dim+5)

    for i in range(n):
        data[i,:]= self.single_sample()
    return data

  def printSample(self,ax=None,alpha=0.3):
      showPlt=False
      if(ax==None):
          showPlt = True
          fig, ax = plt.subplots()
      if self.target_transform == None:
        ax.scatter(self.data[:,0],self.data[:,1],marker='.',alpha=alpha, c=self.data[:,2], cmap=matplotlib.colors.ListedColormap(self.colors))
      else:
        ax.scatter(self.data[:,0],self.data[:,1],marker='.',alpha=alpha, c=self.data[:,2:].argmax(1), cmap=matplotlib.colors.ListedColormap(self.colors))

      if showPlt:
          plt.show()
      else:
          return ax
  def empiricalWeight(self):
    count = torch.zeros(self.distCount)
    for i in range(self.datasetSize):

      if self.target_transform == None:
        count[int(self.data[i,2].item())] += 1
      else:
        count[int(self.data[i,2:].argmax().item())] += 1
    return count/self.datasetSize
  



if __name__ == "__main__":
  ds = CustomSyntheticDataset(datasetSize=10000)
  fig, ax =plt.subplots()
  ax = ds.printSample(ax=ax,alpha=0.2)
  plt.savefig('synthExp2/images/exampleDataset10000')
