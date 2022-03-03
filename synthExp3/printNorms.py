import matplotlib.pyplot as plt
import numpy as np

dist = np.load('synthExp3/dist.npy')
momentumNorm = np.load('synthExp3/boundaries/momentumNorm.npy')

print(dist)
index = np.flip(dist.argsort())
plt.plot(dist[index])
plt.plot(momentumNorm[index]/np.sum(momentumNorm))
plt.xlabel('class number')
plt.ylabel('probability of class')
plt.legend(['class distributions','distributions of norms of W'])
plt.savefig('synthExp3/images/norms',dpi=300)
plt.show()