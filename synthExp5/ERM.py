from dataset import CustomSyntheticDataset
import numpy as np
from lossFunctions import CEL, adapativeLoss, equalisedLoss, logitAdjusted, loss_01
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Lambda


datasetSize = 10000
dataset = CustomSyntheticDataset(dist=[0.99,0.01],datasetSize=datasetSize,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

def test_loop(decisionBoundary, dataset,lossFunction):
  pred = np.zeros(dataset.datasetSize)
  for k in range(dataset.datasetSize):
    if dataset.data[k,0] < decisionBoundary:
      pred[k] = 0
    else:
      pred[k] =1
  #pred = torch.tensor(pred,torch.int64)
  pred = pred.astype(int)
  pred_scatter = torch.zeros((dataset.datasetSize,2))
  for j in range(dataset.datasetSize):
    pred_scatter[j,:] = torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(pred[j],dtype=torch.int64), value=1)
  return lossFunction(pred_scatter,dataset.data[:,1:])


## Find the smallest point for ERM which is decision boundary
inputCount = 200
input = np.linspace(-2,2,inputCount)
loss = np.zeros(inputCount)
#priors = dataset.empiricalWeight()
priors = torch.tensor([0.99,0.01])
lossFunction = lambda x,y : logitAdjusted(x,y,priors)
## let pred be in the form of a step function where boundary is at x
for i in range(inputCount):
  loss[i] = test_loop(input[i],dataset,lossFunction)
# Find the decision boundary which minimises the loss
optimal = input[np.argmin(loss)]
print(optimal)

testDatasetSize = 1000
testDataset=CustomSyntheticDataset(dist=[0.5,0.5],datasetSize=testDatasetSize,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
print(test_loop(optimal,testDataset,loss_01))


plt.plot(input, loss)
plt.savefig('synthExp5/images/test')