import numpy as np
import pickle

## first I will find synth Exp 2 loss
lossComparison = np.load('synthExp2/lossComparisonAccuracy.npy')
print('SYNTH EXP 2 loss mod')
print('Adaptive ',np.mean(lossComparison[:,0]))
print('ERM ',np.mean(lossComparison[:,1]))
print('Equalised ',np.mean(lossComparison[:,2]))
print('Logit Adjusted ',np.mean(lossComparison[:,3]))

## first I will find synth Exp 3 loss
lossComparison = np.load('synthExp3/boundaries/lossComparisonAccuracy.npy')
print('SYNTH EXP 3 loss mod')
print('Adaptive ',np.mean(lossComparison[:,0]))
print('ERM ',np.mean(lossComparison[:,1]))
print('Equalised ',np.mean(lossComparison[:,2]))
print('Logit Adjusted ',np.mean(lossComparison[:,3]))

## synth Exp 2 post hoc
with open('synthExp2/boundaries/postHocMeans.pkl', 'rb') as handle:
    means = pickle.load(handle)

max0 = np.argmax(means[:,0])
max1 = np.argmax(means[:,1])
max2 = np.argmax(means[:,2])
print('Synth Exp2 post hoc')
print("weight normalisation ",means[max0,0])
print("Logit Adjustment ",means[max1,1])
print("re-scaling method ",means[max2,2])

## synth Exp3 post hoc
observations = np.load('synthExp3/observations.npy')
means = np.mean(observations, axis=0)
max0 = np.argmax(means[:,0])
max1 = np.argmax(means[:,1])
max2 = np.argmax(means[:,2])
print('Synth Exp3 post hoc')
print("weight normalisation ",means[max0,0])
print("Logit Adjustment ",means[max1,1])
print("re-scaling method ",means[max2,2])