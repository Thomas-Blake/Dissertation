import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 15}
matplotlib.rc('font', **font)

fig, ax = plt.subplots()

dist = np.load('synthExp3/dist.npy')
dist = -dist
dist.sort()
dist = -dist
ax.plot(np.arange(33), 1000*dist)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Species Rank')
ax.set_ylabel('datapoints per species')


plt.savefig('synthExp3/images/SynthClassDist',dpi=500)