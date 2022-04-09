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
            nn.ReLU()
        )
        self.Wmatrix = torch.normal(mean=torch.zeros((50,11)), std=torch.ones((50,11)))
        self.Wmatrix.requires_grad = True


    def forward(self, x):
        features = self.linear_relu_stack(x)
        return torch.matmul(features, self.Wmatrix)
    




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


def test_loop(dataloader, model, loss_fn,performance=None, tau=torch.tensor(0)):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

  

  

def printDecBoundary(ax,model,detail=1000,color='black',modeltype="torch",distCount=33,a=-10,b=10):
    x1=np.linspace(-10,10,detail+1)
    x2=np.linspace(-10,10,detail+1)
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

    learning_rate = 1e-3


    loss_fn_test = nn.CrossEntropyLoss()



    



    epochs =1000


    observationsAccuracy = np.zeros((20,4))
    observationsLoss = np.zeros((20,4))


    for k in range(20):
        train_dataset = CustomSyntheticDataset(target_transform=Lambda(lambda y: torch.zeros(11, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=10000)
        test_dataset = CustomSyntheticDataset(target_transform=Lambda(lambda y: torch.zeros(11, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),dist=np.ones(11)/11,datasetSize=10000)

        train_dataloader = DataLoader(train_dataset, batch_size=32)
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        priors = train_dataset.empiricalWeight()


        loss_fn = lambda x,y : adapativeLoss(x,y,priors)
        model = NeuralNetwork()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        for t in range(epochs):
            print(f"Epoch {t+1}\n--------------------------loss--0-- k = "+str(k))
            train_loop(train_dataloader, model, loss_fn, optimizer)
        observationsAccuracy[k,0],observationsLoss[k,0]  = test_loop(test_dataloader, model, loss_fn_test)
        #torch.save(model,"./synthExp3/boundaries/nn2-1")

        #ax = train_dataset.printSample(ax)

        model2 = NeuralNetwork()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate)
        for t in range(epochs):
            print(f"Epoch {t+1}\n--------------------------loss--1-- k = "+str(k))
            train_loop(train_dataloader, model2, loss_fn_test, optimizer2)
        observationsAccuracy[k,1],observationsLoss[k,1] = test_loop(test_dataloader, model2, loss_fn_test)

        equalisedLossFn = lambda x,y : equalisedLoss(x,y,priors)
        model3 = NeuralNetwork()
        optimizer3 = torch.optim.SGD(model3.parameters(), lr=learning_rate)
        for t in range(epochs):
            print(f"Epoch {t+1}\n--------------------------loss--2-- k = "+str(k))
            train_loop(train_dataloader, model3, equalisedLossFn, optimizer3)
        observationsAccuracy[k,2],observationsLoss[k,2] = test_loop(test_dataloader, model3, loss_fn_test)

        logitAdjustedFn = lambda x,y : logitAdjusted(x,y,priors)
        model4 = NeuralNetwork()
        optimizer4 = torch.optim.SGD(model4.parameters(), lr=learning_rate)
        for t in range(epochs):
            print(f"Epoch {t+1}\n--------------------------loss--3-- k = "+str(k))
            train_loop(train_dataloader, model4, logitAdjustedFn, optimizer4)
        observationsAccuracy[k,3],observationsLoss[k,3] = test_loop(test_dataloader, model4, loss_fn_test)



    fig, ax = plt.subplots()
    np.save('synthExp2/lossComparisonAccuracy',observationsAccuracy)
    np.save('synthExp2/lossComparisonLoss',observationsLoss)
    ax.boxplot([observationsAccuracy[:,0],observationsAccuracy[:,1],observationsAccuracy[:,2],observationsAccuracy[:,3]])

    




    #plt.show()
    plt.savefig("synthExp2/images/adaptiveERMComparison2",dpi=300)





