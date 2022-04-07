import numpy as np
import matplotlib.pyplot as plt
import itertools
from dataset import CustomSyntheticDataset
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines







def printClassifier(classifier):
    h = 0.02  # step size in the mesh
    # Create color maps
    # cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
    cmap_light = ListedColormap(["blue","red","red","red","red","red","red","red","red","red","red"])
    cmap_bold = ["darkorange", "c", "darkblue"]


    x_min, x_max = -8, 8
    y_min, y_max = -8, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.savefig('testing2')

def printDecBoundary(ax,classifier,detail=1000,color='black'):
    x1=np.linspace(-10,10,detail+1)
    x2=np.linspace(-10,10,detail+1)
    xx1,xx2=np.meshgrid(x1,x2)
    xx1 = np.reshape(xx1,-1)
    xx2 = np.reshape(xx2,-1)
    input = np.vstack((xx1,xx2)).T

    z = classifier.predict(input)


    ## Shape back to original
    xx1 = np.reshape(xx1,(x1.size,x2.size))
    xx2 = np.reshape(xx2,(x1.size,x2.size))
    z= np.reshape(z,(x1.size,x2.size))


    boundary = {}
    for (i,j) in itertools.combinations(range(11), 2):
        boundary[(i,j)] = np.array([[0,0]])
    for i in range(len(x1)):
        for j in range(len(x2)-1):
            if z[i,j] != z[i,j+1]:
                key = (z[i,j],z[i,j+1]) if z[i,j]<z[i,j+1] else (z[i,j+1],z[i,j])
                #print([xx1[i,j],xx2[i,j]])
                boundary[key] = np.append(boundary[key],[[xx1[i,j],xx2[i,j]]],axis=0)
    for j in range(len(x2)):
        for i in range(len(x1)-1):
            if z[i,j] != z[i+1,j]:
                key = (z[i+1,j],z[i,j]) if z[i+1,j]<z[i,j] else (z[i,j],z[i+1,j])
                boundary[key] = np.append(boundary[key],[[xx1[i,j],xx2[i,j]]],axis=0)
    noBoundary = []
    for key in boundary.keys():
        if boundary[key].shape == (1,2):
            noBoundary.append(key)
        else:
            boundary[key] = np.delete(boundary[key],0,0)
    for key in noBoundary:
        boundary.pop(key)

    for value in boundary.values():
        ax.scatter(value[:,0],value[:,1],c=color,s=0.3)
    return ax, boundary


if __name__ == "__main__":
  train_dataset = CustomSyntheticDataset(datasetSize=10000)
  test_dataset = CustomSyntheticDataset(dist=np.ones(11)/11,datasetSize=10000)
  classifier = KNeighborsClassifier(n_neighbors=9)
  classifier.fit(train_dataset.data[:,:2],train_dataset.data[:,2])
  printClassifier(classifier=classifier)
  fig, ax = plt.subplots()
  ax, boundary = printDecBoundary(classifier=classifier,ax=ax,detail=1000)
  train_dataset.printSample(ax=ax,alpha=0.1)
  black_line = mlines.Line2D([], [], color='black',markersize=15, label=r'9-nearest neighbor decision boundary')
  ax.legend(handles=[black_line])

  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  plt.savefig('knearest/9neighborExample',dpi=500)


