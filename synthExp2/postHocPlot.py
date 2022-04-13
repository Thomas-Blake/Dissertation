import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)

with open('synthExp2/boundaries/postHocMeans.pkl', 'rb') as handle:
    means = pickle.load(handle)
  
with open('synthExp2/boundaries/postHocStd.pkl', 'rb') as handle:
    stdDev = pickle.load(handle)

means = 100*means
stdDev = 100*stdDev

fig, ax = plt.subplots()

taus = np.linspace(0,1,50)
max0 = np.argmax(means[:,0])
print(max0)
max1 = np.argmax(means[:,1])
max2 = np.argmax(means[:,2])




ax.plot(taus,means[:,0],color="blue")
ax.plot(taus,means[:,1],color="orange")
ax.plot(taus,means[:,2],color="red")
ax.fill_between(taus, means[:,0]-stdDev[:,0], means[:,0]+stdDev[:,0] ,alpha=0.3, facecolor="blue")
ax.fill_between(taus, means[:,1]-stdDev[:,1], means[:,1]+stdDev[:,1] ,alpha=0.3, facecolor="orange")
ax.fill_between(taus, means[:,2]-stdDev[:,2], means[:,2]+stdDev[:,2] ,alpha=0.3, facecolor="red")
ax.legend(["weight normalisation","Logit Adjustment","re-scaling method"],prop={'size': 14})

plt.scatter(taus[max0], means[max0,0],marker='x',color='black')
ax.text(taus[max0], means[max0,0], "  " + '%.2f' % means[max0,0], transform=ax.transData)
plt.scatter(taus[max1], means[max1,1],marker='x',color='black')
ax.text(taus[max1], means[max1,1], "   " + '%.2f' % means[max1,1], transform=ax.transData)
plt.scatter(taus[max2], means[max2,2],marker='x',color='black')
ax.text(taus[max2], means[max2,2], "  " + '%.2f' % means[max2,2], transform=ax.transData)


ax.yaxis.grid(True)
ax.set_axisbelow(True)


ax.set_xbound([0,1])
ax.set_ybound([60,102])
ax.set_xlabel("tau")
ax.set_ylabel("Test accuracy (%)")




plt.savefig('synthExp2/images/postHocComparison4')
plt.show()