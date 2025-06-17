import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
import pyquickbench


ecc = 0.99
num = 1000

bsa = np.sqrt(1-ecc*ecc)

x = np.empty(num)
y = np.empty(num)

for i in range(num):
    
    t = 2*np.pi * i / (num-1)
    
    x[i] = np.cos(t)
    y[i] = bsa*np.sin(t)
    

plt.plot(x,y)
plt.axis('equal')
plt.show()


