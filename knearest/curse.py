from ast import Lambda
import numpy as np
import matplotlib.pyplot as plt

p = np.arange(1,15)
print(p)
f1 = lambda x: -np.log(100)/np.log(1-0.1**x)
f2 = lambda x: -np.log(100)/np.log(1-0.2**x)
f3 = lambda x: -np.log(100)/np.log(1-0.3**x)
f4 = lambda x: -np.log(100)/np.log(1-0.4**x)


n1 = np.array([f1(xi) for xi in p])
n2 = np.array([f2(xi) for xi in p])
n3 = np.array([f3(xi) for xi in p])
n4 = np.array([f4(xi) for xi in p])

fig, ax = plt.subplots()
ax.plot(p,n1,color='black')
ax.text(p[-1]+0.5,n1[-1],'r=0.1')
ax.plot(p,n2,color='black')
ax.text(p[-1],n2[-1],'r=0.2')
ax.plot(p,n3,color='black')
ax.text(p[-1],n3[-1],'r=0.3')
ax.plot(p,n4,color='black')
ax.text(p[-1],n4[-1],'r=0.4')
ax.set_yscale('log')
ax.set_xbound([0,17])
ax.set_xlabel('p, dimensions')
ax.set_ylabel('minimum n')
plt.savefig('knearest/curse',dpi=500)