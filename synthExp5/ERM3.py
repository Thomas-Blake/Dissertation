from bdb import effective
import torch
from torch import nn
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset import CustomSyntheticDataset
from lossFunctions import CEL, adapativeLoss, equalisedLoss, logitAdjusted, loss_01



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 7),
            # nn.ReLU(),
            # nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(7, 2),
        )

    def forward(self, x):
        x = x.reshape(1,-1).t()

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
    print("Train accuracy: ",100* (correct/size), "% Train numbers: ",correct, " / ",size)


def test_loop(dataloader, model, loss_fn):
    # performance keeps track of the average performance on the tail classes
    size = len(dataloader.dataset)
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
    return correct
  



if __name__ == "__main__":
  train_dataset = CustomSyntheticDataset(dist=[0.99,0.01],target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=100)
  test_dataset = CustomSyntheticDataset(dist=[0.5,0.5],target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),datasetSize=1000)

  train_dataloader = DataLoader(train_dataset, batch_size=8)
  test_dataloader = DataLoader(test_dataset, batch_size=8)

  model = NeuralNetwork()
  learning_rate = 1e-3
  # priors = torch.tensor([0.99,0.01])
  priors = train_dataset.empiricalWeight()

  loss_fn = lambda x,y : logitAdjusted(x,y,priors)
  
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


  for t in range(100):
    print(priors)
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
  print("Done!")

  fig, ax = plt.subplots()
  # ax = train_dataset.printSample(ax)
  # plt.savefig('synthExp5/images/exampleSample')

  input = torch.linspace(-2,2,100)
  out = model(input)
  out = out.detach().numpy()

  ax.plot(input, out[:,0])
  ax.plot(input, out[:,1])
  plt.savefig('synthExp5/images/exampleSample')

