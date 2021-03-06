## This file loads in the boundary pickle files and prints them
import pickle
import matplotlib.pyplot as plt
from dataset import CustomSyntheticDataset
import matplotlib.lines as mlines
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)


with open('./synthExp2/boundaries/balancedNeuralNetBoundary.pkl', 'rb') as handle:
    boundary1 = pickle.load(handle)

with open('./synthExp2/boundaries/normalNeuralNetBoundary.pkl', 'rb') as handle:
    boundary2 = pickle.load(handle)

with open('./synthExp2/boundaries/bayesBalancedBoundary.pkl', 'rb') as handle:
    boundary3 = pickle.load(handle)

with open('./synthExp2/boundaries/bayesNormalBoundary.pkl', 'rb') as handle:
    boundary4 = pickle.load(handle)



fig, ax = plt.subplots()
# for key in boundary1.keys():
#   ax.scatter(boundary1[key][:,0],boundary1[key][:,1],c='black',s=0.3)

# for key in boundary2.keys():
#   ax.scatter(boundary2[key][:,0],boundary2[key][:,1],c='grey',s=0.3)

for key in boundary3.keys():
  ax.scatter(boundary3[key][:,0],boundary3[key][:,1],c='navy',s=0.3)

for key in boundary4.keys():
  ax.scatter(boundary4[key][:,0],boundary4[key][:,1],c='green',s=0.3)

ds = CustomSyntheticDataset(datasetSize=10000)
ds.printSample(ax,alpha=0.1)

# labels = ['Neural Net Balanced loss','Neural Net cross-entropy loss','Bayes Balanced loss','Bayes cross-entropy loss']
# legend = ax.legend(loc="upper left", fontsize=7, markerscale=10, labels=labels)
# handles = legend.legendHandles

# colors = ['black','grey','blue','green']
# for i, handle in enumerate(handles):
#     handle.set_facecolor(colors[i])

# black_line = mlines.Line2D([], [], color='black',markersize=15, label='Neural Net Balanced loss')
# grey_line = mlines.Line2D([], [], color='grey',markersize=15, label='Neural Net cross-entropy loss')
blue_line = mlines.Line2D([], [], color='navy',markersize=15, label=r'$f_{TEST}^*$')
green_line = mlines.Line2D([], [], color='green',markersize=15, label=r'$f_{TRAIN}^*$')

# ax.legend(handles=[black_line,grey_line,blue_line,green_line])
ax.legend(handles=[blue_line,green_line])

ax.set_xlabel('x1')
ax.set_ylabel('x2')

ax.set_xbound([-8,8])
ax.set_ybound([-8,8])





plt.show()
plt.savefig('synthExp2/images/bayesDecisionBoundaries',dpi=500)