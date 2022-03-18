from bdb import effective
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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 33),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        print(logits.shape)
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
    print("Train accuracy: ",100* (correct/size), "% Train numbers: ",correct, " / ",size)


def test_loop(dataloader, model, loss_fn,performance=None):
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
    if performance:
        performance.append(tail_correct/tail_size)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  

  

def printDecBoundary(ax,model,detail=1000,color='black',modeltype="torch",distCount=33,a=-10,b=10):
    x1=np.linspace(-10,20,detail+1)
    x2=np.linspace(-15,10,detail+1)
    xx1,xx2=np.meshgrid(x1,x2)
    xx1 = np.reshape(xx1,-1)
    xx2 = np.reshape(xx2,-1)
    input = np.vstack((xx1,xx2)).T
    if(modeltype == 'torch'):
        input = torch.tensor(input,dtype=torch.float,requires_grad=False)
        z=np.argmax(model(input).detach().numpy(),axis=1)
    else:
        # used for finding Bayes decision boundary
        ## z=np.argmax(model(input[:,0],input[:,1]),axis=1)
        z = model(input[:,0],input[:,1])


    ## Shape back to original
    xx1 = np.reshape(xx1,(x1.size,x2.size))
    xx2 = np.reshape(xx2,(x1.size,x2.size))
    z= np.reshape(z,(x1.size,x2.size))


    boundary = {}
    for (i,j) in itertools.combinations(range(distCount), 2):
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

    for value in boundary.values():
        ax.scatter(value[:,0],value[:,1],c=color,s=0.3)
    return ax, boundary


if __name__ == "__main__":

    train_dataset = CustomSyntheticDataset(dist=np.load('synthExp3/dist.npy'),target_transform=Lambda(lambda y: torch.zeros(33, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=10000)
    test_dataset = CustomSyntheticDataset(dist=np.ones(33)/33,target_transform=Lambda(lambda y: torch.zeros(33, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=1000)


    # train_dataset = CustomSyntheticDataset(datasetSize=10000)
    # test_dataset = CustomSyntheticDataset(datasetSize=1000)
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    model = NeuralNetwork()


    learning_rate = 1e-3




    balancedLoss = False
    if(balancedLoss):
        classDist = train_dataset.empiricalWeight()
        weights = torch.zeros(train_dataset.distCount)
        for i in range(train_dataset.distCount):
            if classDist[i] != 0:
                weights[i] = 1/classDist[i]

        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        # Initialize the loss function
        # this loss function includes softmax
        loss_fn = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    epochs =1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    # torch.save(model,"./synthExp1/normalNeuralNet")

    fig, ax = plt.subplots()
    ax, boundary = printDecBoundary(ax,model,detail=100)
    ax = train_dataset.printSample(ax)
    #plt.savefig('synthExp3/images/neuralNet2')

    ## Save to boundary
    if False:
        if balancedLoss:
            with open('./synthExp3/balancedNeuralNetBoundary.pkl', 'wb') as f:
                pickle.dump(boundary, f)
        else:
            with open('./synthExp3/normalNeuralNetBoundary.pkl', 'wb') as f:
                pickle.dump(boundary, f)


    #train_dataset.printSample(ax)

    plt.show()





