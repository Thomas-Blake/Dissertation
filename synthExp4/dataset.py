from dis import dis
import torch
from torch import nn
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import itertools
import matplotlib

def distCreater():
  dist=np.ones(33)
  for i in range(3):
    dist[11*i] = np.random.uniform(100,200)
  for j in range(3):
    for i in range(10):
      dist[1+i+11*j] = np.random.uniform(0,10)
  return dist/np.sum(dist)

def expDist(mu = 0.9, imbalanceFactor = None):
  if(imbalanceFactor):
    mu = imbalanceFactor**(-1/32)
  dist = 2*np.ones(33)
  for i in range(3,33):
    dist[i] = mu**i
  # normalise
  dist=dist/np.sum(dist)
  ## change the order so that tail, head classes are in right place
  
  #plt.plot(dist)
  #plt.show()
  np.random.shuffle(dist[3:])
  np.save("synthExp3/dist",dist)
  plt.plot(dist)
  plt.savefig('synthExp3/distFigure')
  return dist




class CustomSyntheticDataset(Dataset):
  def __init__(self, datasetSize=1000,dist=np.ones(15)/15,target_transform=None):
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

    for i in range(3):
      for j in range(5):
        #print(5*np.cos(2*np.pi*i/3)+np.cos(2*np.pi*j/5),5*np.sin(2*np.pi*i/3)+np.sin(2*np.pi*j/5))
        #plt.scatter(4*np.cos(2*np.pi*i/3)+2*np.cos(2*np.pi*j/5),4*np.sin(2*np.pi*i/3)+2*np.sin(2*np.pi*j/5),c="black")
        self.mus.append(torch.tensor([4*np.cos(2*np.pi*i/3)+2*np.cos(2*np.pi*j/5),4*np.sin(2*np.pi*i/3)+2*np.sin(2*np.pi*j/5)],dtype=torch.float))
        self.sigmas.append(torch.tensor([[0.4,0.],[0.,0.4]]))
        self.distributions.append(MultivariateNormal(self.mus[5*i+j], covariance_matrix=self.sigmas[5*i+j]))
        self.colors.append("black")



    self.distCount = len(self.distributions)
    self.colors = ["lightcoral","indianred","brown","firebrick","maroon","olive","forestgreen","limegreen","green","darkseagreen","skyblue","cornflowerblue","blue","navy","steelblue"]

    if target_transform == None:
      self.data=torch.zeros(self.datasetSize,self.dim+1)
    else:
      self.data=torch.zeros(self.datasetSize,self.dim+33)

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
      showPlt=True
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
      #plt.savefig('synthExp3/test')
  def empiricalWeight(self):
    count = torch.zeros(self.distCount)
    for i in range(self.datasetSize):

      if self.target_transform == None:
        count[int(self.data[i,2].item())] += 1
      else:
        count[int(self.data[i,2:].argmax().item())] += 1
    return count/self.datasetSize
  
if __name__ == "__main__":
  data = CustomSyntheticDataset()
  data.printSample()

