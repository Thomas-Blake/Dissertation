import torch
from torch import nn
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import itertools
import matplotlib
from dataset import CustomSyntheticDataset
import pickle
import matplotlib.lines as mlines



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 11),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct =0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Train accuracy: ",100* (correct/size), " Train numbers: ",correct, " / ",size)


def test_loop(dataloader, model, loss_fn,performance):
    # performance keeps track of the average performance on the tail classes
    size = len(dataloader.dataset)
    tail_size = size - train_dataset.empiricalWeight()[0]
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    tail_correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            tail_correct += ((pred.argmax(1) == y.argmax(1)) & (y.argmax(1) != 0)).type(torch.float).sum().item()


    test_loss /= num_batches
    correct /= size
    performance.append(tail_correct/tail_size)
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
  for (i,j) in itertools.combinations(range(11), 2):
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
  return ax, boundary

  




# if __name__ == "__main__":

#     train_dataset = CustomSyntheticDataset(target_transform=Lambda(lambda y: torch.zeros(11, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=1000)
#     test_dataset = CustomSyntheticDataset(dist = np.ones(11)/11,target_transform=Lambda(lambda y: torch.zeros(11, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=100)


#     # train_dataset = CustomSyntheticDataset(datasetSize=10000)
#     # test_dataset = CustomSyntheticDataset(datasetSize=1000)

#     train_dataloader = DataLoader(train_dataset, batch_size=64)
#     test_dataloader = DataLoader(test_dataset, batch_size=64)

#     model = NeuralNetwork()


#     learning_rate = 1e-3
#     batch_size = 1

#     classDist = train_dataset.empiricalWeight()
#     weights = torch.zeros(train_dataset.distCount)

#     for i in range(train_dataset.distCount):
#         if classDist[i] != 0:
#             weights[i] = 1/classDist[i]

#     # Initialize the loss function
#     # this loss function includes softmax
#     balanced=False
#     if balanced:
#         loss_fn = nn.CrossEntropyLoss(weight=weights)
#     else:
#         loss_fn = nn.CrossEntropyLoss()

#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#     epochs = 4000
#     for t in range(epochs):
#         print(f"Epoch {t+1}\n-------------------------------")
#         train_loop(train_dataloader, model, loss_fn, optimizer)
#         test_loop(test_dataloader, model, loss_fn)
#     print("Done!")
#     # torch.save(model,"./synthExp1/normalNeuralNet")

#     fig, ax = plt.subplots()
#     ax, boundary = printDecBoundary(ax,model,detail=1000)

#     if(balanced):
#         with open('./synthExp1/balancedNeuralNetBoundary.pkl', 'wb') as f:
#             pickle.dump(boundary, f)
#     else:
#         with open('./synthExp1/normalNeuralNetBoundary.pkl', 'wb') as f:
#             pickle.dump(boundary, f)




#     #test_dataset.printSample(ax)
#     test_dataset.printSample(ax)


#     plt.show()








if __name__ == "__main__":

    train_dataset = CustomSyntheticDataset(target_transform=Lambda(lambda y: torch.zeros(11, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=5000)
    test_dataset = CustomSyntheticDataset(dist = np.ones(11)/11,target_transform=Lambda(lambda y: torch.zeros(11, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=500)


    # train_dataset = CustomSyntheticDataset(datasetSize=10000)
    # test_dataset = CustomSyntheticDataset(datasetSize=1000)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model1 = NeuralNetwork()


    learning_rate = 1e-3
    batch_size = 64

    classDist = train_dataset.empiricalWeight()
    weights = torch.zeros(train_dataset.distCount)

    for i in range(train_dataset.distCount):
        if classDist[i] != 0:
            weights[i] = 1/classDist[i]

    # Initialize the loss function
    # this loss function includes softmax

    loss_fn = nn.CrossEntropyLoss(weight=weights)


    optimizer = torch.optim.SGD(model1.parameters(), lr=learning_rate)


    epochs = 700
    balanced_performance = []
    uniform_performance = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model1, loss_fn, optimizer)
        test_loop(test_dataloader, model1, loss_fn,balanced_performance)
    # torch.save(model,"./synthExp1/normalNeuralNet")
    epochs = 700
    loss_fn = nn.CrossEntropyLoss()
    model2 = NeuralNetwork()
    optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model2, loss_fn, optimizer)
        test_loop(test_dataloader, model2, loss_fn,uniform_performance)
    print("Done!")
    # torch.save(model,"./synthExp1/normalNeuralNet")
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    black_line = mlines.Line2D([], [], color='black',markersize=15, label='Neural Net Balanced loss')
    grey_line = mlines.Line2D([], [], color='grey',markersize=15, label='Neural Net cross-entropy loss')
    ax1.legend(handles=[black_line,grey_line])
    ax1, boundary = printDecBoundary(ax1,model1,detail=1000,color="black")
    ax1, boundary = printDecBoundary(ax1,model2,detail=1000,color="grey")
    test_dataset.printSample(ax1)

    fig2, ax2 = plt.subplots()
    ax2.plot(balanced_performance)
    ax2.plot(uniform_performance)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy on tail classes")
    ax2.set_xscale("log")
    plt.show()






    #test_dataset.printSample(ax)
    #test_dataset.printSample(ax)


    #plt.show()







