import matplotlib.pyplot as plt
import numpy as np
import matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)

observations = 100*np.load('synthExp3/boundaries/lossComparisonAccuracy.npy')

fig, ax = plt.subplots()
bplot = ax.boxplot([observations[:,0],observations[:,1],observations[:,2],observations[:,3]],patch_artist=True)
colors = ['orange','cornflowerblue','green','red']
for patch, color in zip(bplot['boxes'], colors):
  patch.set_facecolor(color)

plt.setp(bplot['medians'], color='black')
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim([70,100])
ax.set_xticklabels(["Adaptive","ERM","Equalised","Logit adjusted"])

plt.savefig("synthExp3/images/lossComparison2",dpi=500)
