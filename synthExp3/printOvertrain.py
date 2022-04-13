from bdb import effective
from tokenize import group
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
import matplotlib

font = {'size'   : 15}
matplotlib.rc('font', **font)

observationsTrain = np.load('synthExp3/overfitTrain.npy')
observationsTest = np.load('synthExp3/overfitTest.npy')


observationsTrainMean = observationsTrain.mean(axis=0)
observationsTestMean = observationsTest.mean(axis=0)


fig, ax = plt.subplots()
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.bar(np.arange(7)-0.15, 100*observationsTrainMean,width=0.3)
ax.bar(np.arange(7)+0.15, 100*observationsTestMean,width=0.3)
ax.set_xlabel("Class group")
ax.set_ylabel("Accuracy %")
ax.legend(["train","test"],framealpha=1,loc='lower left')
plt.savefig('synthExp3/images/classAccuracy9',dpi=500)