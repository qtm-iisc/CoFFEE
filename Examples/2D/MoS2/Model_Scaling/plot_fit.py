#!/usr/bin/python
import sys,string
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
import numpy as np

fig, ax = plt.subplots()

# Values of alpha:
alpha = np.array([4,5,6,8,10,20,40,80])

# Corresponding model energies:
En = np.array([0.421,0.464,0.490,0.516,0.529,0.555,0.588,0.618])

one_alpha = 1./alpha

# Compute the fifth-order polynomial fit
P = np.polyfit(one_alpha,En,5)
print("Polynomial coefficients:", P)
print("Isolated model energy: %2.3f"%( P[5]))

En_f = []
one_alpha_f = np.arange(0.,0.27,0.004)
for x in one_alpha_f:
  En_f.append(P[5] + P[4]*x + P[3]*x**2 + P[2]*x**3 + P[1]*x**4 + P[0]*x**5)

# Scatter the model energy values
ax.scatter(one_alpha,En,lw = 2.,color = 'navy',marker='o',s=280)

# Plot the fitting polynomial
ax.plot(one_alpha_f,En_f,lw = 3.8,color = 'navy')
ax.set_xlim(0,0.27)
ax.set_ylim(0.4,0.8)
ax.set_xlabel(r"$1/\alpha$",fontsize = 40)
ax.set_ylabel(r"$\mathrm{E}_{-1}^{\mathrm{m,per}} \mathrm{(eV)}$",fontsize = 40)
ax.set_yticks(np.arange(0.4,0.82,0.1))
ax.tick_params(labelsize=26,width=2.5,length=6)
plt.tight_layout()
plt.savefig("Scaling")
plt.show()
