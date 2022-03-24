import numpy as np
observation = np.load('synthExp3/observations.npy')
#observation = np.mean(observation,axis=0)
print(observation[9,:,:])