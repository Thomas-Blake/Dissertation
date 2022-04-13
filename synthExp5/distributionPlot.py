from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

font = {'size'   : 15}
matplotlib.rc('font', **font)


x = np.linspace(-2.5,2.5,100)
dist1 = (2 *np.pi)**(-0.5)*np.exp(-0.5*(x+2)**2)
dist2 = (2 *np.pi)**(-0.5)*np.exp(-0.5*(x-2)**2)

fig, ax = plt.subplots()

ax.set_xticks([-2,0,2])


ax.plot([0,0],[0,0.5],color='black',linewidth=3)
ax.plot([0,0],[0,0.5],'--',color='orange',linewidth=3)


ax.plot([0.25*np.log(99),0.25*np.log(99)],[0,0.5],color='grey',linewidth=3)

ax.plot([0.25*np.log(1.99/1.01),0.25*np.log(1.99/1.01)],[0,0.5],color='purple',linewidth=3)

#Adaptive
ax.plot([1.08,1.08],[0,0.5],color='green',linewidth=3)
legend1 = ax.legend([r'$f_{TEST}^*$','Logit Adjusted loss',r'$f_{TRAIN}^*$','Equalised','Adaptive'],title="Decision Boundaries",prop={'size': 11})
ax.fill_between(x, 0, dist1,color='blue',alpha=0.3)
ax.fill_between(x, 0, dist2,color='red',alpha=0.3)

red_patch = mpatches.Patch(color='red', label='Y=2',alpha=0.3)
blue_patch = mpatches.Patch(color='blue', label='Y=1',alpha=0.3)
ax.legend(handles=[red_patch,blue_patch], title="Distributions",prop={'size': 11})
#ax.legend(['Y=1','Y=2'])
plt.gca().add_artist(legend1)


ax.set_xbound([-2.5,2.5])
ax.set_ybound([0,0.7])
ax.set_xlabel('x')
plt.rcParams['text.usetex'] = True
ax.set_ylabel('$f_{X|Y=y}(x)$')

plt.savefig('synthExp5/images/dist')