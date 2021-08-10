#!/usr/bin/env python

import sys,string
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
import numpy as np

fig, ax = plt.subplots()

# Values of alpha:
alpha = np.array([6,8,10,15,20,30])

# Corresponding model energies:
En = np.array([1.457,1.589,1.651,1.7403,1.7882,1.8158])

one_alpha = 1./alpha

# Compute the fifth-order polynomial fit
P = np.polyfit(one_alpha,En,3)
print("Polynomial coefficients:", P)
print("Isolated model energy: %2.3f"%( P[3]))

En_f = []
one_alpha_f = np.arange(0.,0.27,0.004)
for x in one_alpha_f:
  En_f.append(P[3] + P[2]*x + P[1]*x**2 + P[0]*x**3)

# Scatter the model energy values
ax.scatter(one_alpha,En,lw = 2.,color = 'navy',marker='o',s=280)

# Plot the fitting polynomial
ax.plot(one_alpha_f,En_f,lw = 3.8,color = 'red')
ax.set_xlim(0,0.18)
ax.set_ylim(1.4,2.0)
ax.set_xlabel(r"$1/\alpha$",fontsize = 40)
ax.set_ylabel(r"$\mathrm{E}_{-1}^{\mathrm{m,per}} \mathrm{(eV)}$",fontsize = 40)
ax.set_xticks(np.arange(0.0,0.18,0.04))
ax.tick_params(labelsize=26,width=2.5,length=6)
plt.tight_layout()
plt.savefig("Scaling")
plt.show()
