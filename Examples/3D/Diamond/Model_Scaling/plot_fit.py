#!/usr/bin/python
import sys, string
import numpy as np
import matplotlib.pyplot as plt

def compute_fit(C_u,L2,L3,L4):
  #
  # Computes the fitting polynomial.
  # C_u takes the 3 model energies
  # L2, L3 and L4 are 1/\Omega^{1/3} for the correspoding cells. 
  #
  A = np.array([[ L2, L2**3, 1.], [  L3, L3**3, 1.] , [  L4, L4**3, 1. ] ])
  A_inv =  np.linalg.inv(A)
  X_u = np.dot(A_inv,C_u)
  return X_u

alat = 6.6486*0.52918
RYD = 13.605698

# The Model energies go in here:
E_m = np.array([1.07438,1.2549,1.38])

# 1/\Omega^{1/3} for 4x4x4, 5x5x5 and 6x6x6 cells. 
one_by_V = np.array([1./(4.**3),1/(5.**3),1/(6.**3)])
one_by_A = (one_by_V)**(1/3.)*(1/alat)

# Compute the fit: p(\Omega) = f_1 + f_2/(\Omega^(1/3)) + f_3/(\Omega)
X = compute_fit([E_m[0],E_m[1],E_m[2]],one_by_A[0],one_by_A[1],one_by_A[2])

# Use the coefficient obtained above to generate the fitting curve
Linv = np.arange(0,0.15,0.005)
Y = []
for x in Linv:
  Y.append(X[0]*x + X[1]*x**3 + X[2])

E_m_iso = X[2]

print("E_m_iso:", X[2])
print("Fitting parameters (f_1,f_2,f_3):", X[2],X[0],X[1])
fig,(ax) = plt.subplots()
# Scatter the model energies
ax.scatter(one_by_A, E_m,lw = 2.,color = 'k',marker='o',s=250,label = "Uncorrected")
# Plot the fitted curve
ax.plot(Linv,Y,color = 'r', lw = 3.8)
ax.set_ylabel(r'$\mathrm{E}_{-2}^{\mathrm{m,per}}$ (eV)',fontsize = 32)
ax.set_xlabel(r'$\Omega^{\mathrm{-1/3}}$ ($\AA^{-1}$)',fontsize = 32)
ax.tick_params(labelsize = 24, width = 2.5,length = 7)
ax.set_xticks([0,0.02,0.04,0.06, 0.08])
ax.set_xlim(0,0.08)
ax.set_ylim(0.8,2.2)
ax.set_title("Sigma = 2.614",fontsize=28)
plt.tight_layout()
plt.savefig("E_iso.png")
plt.show()
