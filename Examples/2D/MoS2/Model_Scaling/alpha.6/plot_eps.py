import numpy as np
import matplotlib.pyplot as plt

a = 35.436*0.52918
eps1 = np.load("epsZ_a1.npy")
eps2 = np.load("epsZ_a2.npy")
eps3 = np.load("epsZ_a3.npy")
Z = np.linspace(0.,a,len(eps1))

fig, ax = plt.subplots()
plt.plot(Z,eps1,lw = 3.8,color = 'k',label = "Along a1")
plt.plot(Z,eps2,lw = 3.8,color = 'red',linestyle = '--',label = "Along a2")
plt.plot(Z,eps3,lw = 3.8,color = 'green',label =  "Along a3")
plt.legend(fontsize = 22)
plt.xlim(0,a)
ax.set_ylabel(r"$\varepsilon(z)$",fontsize = 32)
ax.set_xlabel(r"$z$ ($\AA$)",fontsize = 32)
ax.tick_params(labelsize = 19, width = 2.5,length = 7)
plt.tight_layout()
plt.savefig("Eps_profile.png")
plt.show()
