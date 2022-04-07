import matplotlib.pyplot as plt
import numpy as np

observations = np.load('synthExp2/lossComparison.npy')

fig, ax = plt.subplots()
bplot = ax.boxplot([observations[:,0],observations[:,1],observations[:,2],observations[:,3]],patch_artist=True)
colors = ['orange','cornflowerblue','green','red']
for patch, color in zip(bplot['boxes'], colors):
  patch.set_facecolor(color)

plt.setp(bplot['medians'], color='black')
ax.set_ylabel("Test accuracy")
ax.set_ylim([0.3,1])
ax.set_xticklabels(["Adaptive","ERM","Equalised","Logit adjusted"])

plt.savefig("synthExp2/images/lossComparison",dpi=300)