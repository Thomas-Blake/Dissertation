import numpy as np
import matplotlib.pyplot as plt

observation = np.load('knearest/observation.npy')

mean = np.mean(observation,axis=0)
stdDev = np.std(observation,axis=0)
fig,ax = plt.subplots()

bpl = ax.boxplot(observation,positions=np.arange(1,11))

ax.set_xlabel('k value')
ax.set_ylabel('Accuracy on test set')

plt.savefig('knearest/changingNeighbourNum',dpi=500)