import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


observation_empirical = np.load('synthExp5/empiricalERM2.npy')
observation_true = np.load('synthExp5/trueERM2.npy')



fig, ax = plt.subplots()
#bpl = ax.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6)
flierprops = dict(marker='o', markerfacecolor='red',alpha=0.5,markeredgewidth=0.0)
for i in range(6):
  bpl = ax.boxplot(observation_empirical[:,i],positions=[i+1-0.15],widths=0.2,patch_artist=True, flierprops=flierprops)
  bpl['boxes'][0].set_facecolor('red')
  plt.setp(bpl['medians'], color='black')
flierprops = dict(marker='o', markerfacecolor='blue',alpha=0.5,markeredgewidth=0.0)
for i in range(6):
  bpl = ax.boxplot(observation_true[:,i],positions=[i+1+0.15],widths = 0.2,patch_artist=True,flierprops=flierprops)
  bpl['boxes'][0].set_facecolor('blue')
  bpl['fliers'][0].set_color('red')
  plt.setp(bpl['medians'], color='black')

print(bpl['fliers'][0])
ax.set_xticks(np.arange(1,7), [200,400,800,1600,3200,6400])
ax.set_xlabel('Size of train dataset')
ax.set_ylabel('Accuracy on test dataset')
ax.set_ybound([0.9,1])

# blue_line = mpatches.Rectangle([], [], color='blue',markersize=15, label='Bayes Balanced loss')
# green_line = mpatches.Rectangle([], [], color='green',markersize=15, label='Bayes cross-entropy loss')
red_patch = mpatches.Patch(facecolor='red',label='Empirical prior estimate')
blue_patch = mpatches.Patch(facecolor='blue',label='True prior')

ax.legend(handles=[red_patch,blue_patch],prop={'size': 12})


plt.savefig('synthExp5/images/ERMvsTruePrior',dpi=500)