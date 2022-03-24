import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('synthExp3/postHocMeans.pkl', 'rb') as handle:
    means = pickle.load(handle)
  
with open('synthExp3/postHocStd.pkl', 'rb') as handle:
    stdDev = pickle.load(handle)

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
ax.legend(["weight normalization 1","logit adjustment","weight normalization 2"])

plt.scatter(taus[max0], means[max0,0],marker='x',color='black')
ax.text(taus[max0], means[max0,0], "  " + '%.4f' % means[max0,0], transform=ax.transData)
plt.scatter(taus[max1], means[max1,1],marker='x',color='black')
ax.text(taus[max1], means[max1,1], "   " + '%.4f' % means[max1,1], transform=ax.transData)
plt.scatter(taus[max2], means[max2,2],marker='x',color='black')
ax.text(taus[max2], means[max2,2], "  " + '%.4f' % means[max2,2], transform=ax.transData)

ax.set_xbound([0,1])
ax.set_ybound([0,1.02])
ax.set_xlabel("tau")
ax.set_ylabel("test accuracy")




plt.savefig('testing')
plt.show()