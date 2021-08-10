import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt("plavg_a3.plot")
plt.plot(A[:,0],A[:,1])
plt.show()
