import torch
from torch import nn
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import itertools


class CustomSyntheticDataset(Dataset):
  def __init__(self, datasetSize=100000,dist=(0.2,0.2,0.2,0.2,0.2),target_transform=None):
    self.target_transform = target_transform
    self.datasetSize = datasetSize
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

    self.data=np.zeros((self.datasetSize,self.dim+5))
    for i in range(self.datasetSize):
      self.data[i,:]= self.single_sample()


  def __len__(self):
    return self.datasetSize

    
  def single_sample(self):
    choice = np.random.choice(len(self.dist),p=self.dist)
    if self.target_transform:
      return np.append(self.distributions[choice].sample(),self.target_transform(choice))
    else:
      return np.append(self.distributions[choice].sample(),choice)

  def __getitem__(self, idx):
    return torch.tensor(self.data[idx,0:2],dtype=torch.float),torch.tensor(self.data[idx,2:],dtype=torch.float)




class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        print(pred)
        print(pred.dtype)
        print(y)
        print(y.dtype)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  
def printDecBoundary(ax,model,detail=100,color='black'):
  x1=np.linspace(-10,10,detail+1)
  x2=np.linspace(-10,10,detail+1)
  xx1,xx2=np.meshgrid(x1,x2)
  z = np.zeros(xx1.shape)
  for i in range(detail+1):
    if(i % 10 ==0):
      print(i)
    for j in range(detail+1):
        ##z[i,j] = self.makePrediction(xx1[i,j],xx2[i,j],balancedLoss)
        z[i,j] = model(torch.tensor([[xx1[i,j],xx2[i,j]]],dtype=torch.float,requires_grad=False)).argmax().detach().numpy()


  boundary = {}
  print(z)
  for (i,j) in itertools.combinations(range(5), 2):
      boundary[(i,j)] = np.array([[0,0]])
  for i in range(len(x1)):
      for j in range(len(x2)-1):
          if z[i,j] != z[i,j+1]:
              print(i,j)
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
  return ax

  




if __name__ == "__main__":

  train_dataset = CustomSyntheticDataset(target_transform=Lambda(lambda y: torch.zeros(5, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=100000)
  test_dataset = CustomSyntheticDataset(target_transform=Lambda(lambda y: torch.zeros(5, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=10000)

  train_dataloader = DataLoader(train_dataset, batch_size=64)
  test_dataloader = DataLoader(test_dataset, batch_size=64)

  model = NeuralNetwork()


  learning_rate = 1e-3
  batch_size = 64
  epochs = 5

  # Initialize the loss function
  loss_fn = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


  epochs = 2
  for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train_loop(train_dataloader, model, loss_fn, optimizer)
      test_loop(test_dataloader, model, loss_fn)
  print("Done!")

  print(model(torch.tensor([[0,0]],dtype=torch.float,requires_grad=False)).argmax())
  fig, ax = plt.subplots()
  printDecBoundary(ax,model,detail=100)
  ## Plot Scatter
  plt.scatter(test_dataset.data[:,0],test_dataset.data[:,1],c=np.argmax(test_dataset.data[:,2:],axis=1),marker='x',s=0.4,alpha=0.4)

  plt.show()



