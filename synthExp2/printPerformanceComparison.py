## look at how accuracy compares for balanced and standard ERM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)


trackerTestBalanced = np.load('synthExp2/boundaries/accuracyTrackerTestBalanced.npy')
trackerTestNormal = np.load('synthExp2/boundaries/accuracyTrackerTestNormal.npy')
trackerTrainBalanced = np.load('synthExp2/boundaries/accuracyTrackerTrainBalanced.npy')
trackerTrainNormal  = np.load('synthExp2/boundaries/accuracyTrackerTrainNormal.npy')

epochs = 5000
fig, ax = plt.subplots()

ax.plot(np.arange(epochs),100*trackerTestBalanced,'--', linewidth=2,label=r'Test - trained on $L^{BAL}$',color='green')
ax.plot(np.arange(epochs),100*trackerTestNormal,'--',label=r'Test - trained on $L^{CE}$',color='navy')
ax.plot(np.arange(epochs),100*trackerTrainBalanced,label=r'Train - trained on $L^{BAL}$',color='green')
ax.plot(np.arange(epochs),100*trackerTrainNormal,label=r'Train - trained on $L^{CE}$',color='navy')

ax.set_xlabel('Epoch')
ax.set_xscale('log')
ax.set_ylabel('Accuracy (%)')
ax.legend(loc=4,prop={'size': 11})
plt.savefig('synthExp2/images/performanceComparison',dpi=500)