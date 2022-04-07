from contextlib import redirect_stderr
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
from torch import linalg as LA
from lossFunctions import CEL, adapativeLoss, equalisedLoss, logitAdjusted

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
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct =0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        #adapativeLoss(pred,y,priors)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Train accuracy: ",100* (correct/size), "% Train numbers: ",correct, " / ",size)


def test_loop(dataloader, model, loss_fn,zeros,performance=None, tau=torch.tensor(0)):
    # performance keeps track of the average performance on the tail classes
    size = len(dataloader.dataset)
    tail_size = size - train_dataset.empiricalWeight()[0]
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    tail_correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            ## we predict -inf for zero values
            minCol = torch.ones(y.shape[0])*torch.finfo(torch.float32).min
            for col in zeros:
                pred[:,col] = minCol

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            tail_correct += ((pred.argmax(1) == y.argmax(1)) & (y.argmax(1) != 0)).type(torch.float).sum().item()


    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct 

  


if __name__ == "__main__":



    train_datasizes = [2500, 5000,10000,20000,40000]
    observationCount = 10
    observations_empirical = np.zeros((observationCount,len(train_datasizes)))
    observations_true = np.zeros((observationCount,len(train_datasizes)))
    fig, ax = plt.subplots()
    #colors= ['blue','red','green']
    #epochs = 10
    epochs = 1000

    for i in range(len(train_datasizes)):
      for k in range(observationCount):
        train_dataset = CustomSyntheticDataset(dist=np.load('synthExp3/dist.npy'),target_transform=Lambda(lambda y: torch.zeros(33, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=train_datasizes[i])
        test_dataset = CustomSyntheticDataset(dist=np.ones(33)/33,target_transform=Lambda(lambda y: torch.zeros(33, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=10000)

        train_dataloader = DataLoader(train_dataset, batch_size=32)
        test_dataloader = DataLoader(test_dataset, batch_size=32)

        model = NeuralNetwork()
        learning_rate = 1e-3
        priors = train_dataset.empiricalWeight()
        zeros = []
        for j in range(33):
            if priors[j] ==0:
                print("we have a 0")
                zeros.append(j)
        loss_fn = lambda x,y : logitAdjusted(x,y,priors)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


        for t in range(epochs):
            print(f"Epoch {t+1}\n----------Empirical "+str(i)+" ---- observation Count --- "+str(k))
            train_loop(train_dataloader, model, loss_fn, optimizer)
        observations_empirical[k,i] = test_loop(test_dataloader, model, loss_fn,zeros)
        print("Done!")

    
    np.save('synthExp3/boundaries/priors_empirical',observations_empirical)
    observations_empirical = np.mean(observations_empirical,axis=0)

    for i in range(len(train_datasizes)):
      for k in range(observationCount):
        train_dataset = CustomSyntheticDataset(dist=np.load('synthExp3/dist.npy'),target_transform=Lambda(lambda y: torch.zeros(33, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=train_datasizes[i])
        test_dataset = CustomSyntheticDataset(dist=np.ones(33)/33,target_transform=Lambda(lambda y: torch.zeros(33, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=10000)

        train_dataloader = DataLoader(train_dataset, batch_size=32)
        test_dataloader = DataLoader(test_dataset, batch_size=32)

        model = NeuralNetwork()
        learning_rate = 1e-3
        priors = torch.tensor(np.load('synthExp3/dist.npy'))
        loss_fn = lambda x,y : logitAdjusted(x,y,priors)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



        for t in range(epochs):
            print(f"Epoch {t+1}\n----------True "+str(i)+" ---- observation Count --- "+str(k))
            train_loop(train_dataloader, model, loss_fn, optimizer)
        observations_true[k,i] = test_loop(test_dataloader, model, loss_fn,[])
        print("Done!")
        # input = torch.linspace(-2,2,100)
        # out = model(input)
        # out = out.detach().numpy()

        # ax.plot(input, out[:,0],color=colors[i])
        # ax.plot(input, out[:,1],color=colors[i])
    
    np.save('synthExp3/boundaries/priors_true',observations_true)
    observations_true = np.mean(observations_true,axis=0)
    

    ax.plot(train_datasizes,observations_empirical)
    ax.plot(train_datasizes,observations_true)
    ax.legend(["empirical","true"])

    plt.savefig('synthExp3/images/ERMPriorvsTrue')




