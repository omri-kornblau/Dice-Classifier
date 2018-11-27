import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
a = np.random.randn(1000)
std2 = np.std(a)
mean2 = np.mean(a)
b = (1/(np.sqrt(2*3.14159)*std2))*np.exp((-1*(a-mean2)**2)/(2*std2**2))

c = np.random.rand(1000)
std = np.std(c)
mean = np.mean(c)
d = (1/(np.sqrt(2*3.14159)*std))*np.exp((-1*(c-mean)**2)/(2*std**2))
f, (ax, bx) = plt.subplots(2)

ax.grid()
ax.scatter(a, b, s=1)
ax.axvline(x=std2)
ax.axvline(x=mean2)
bx.grid()
bx.scatter(c, d, s=1)
bx.axvline(x=std)
bx.axvline(x=mean)
plt.show()