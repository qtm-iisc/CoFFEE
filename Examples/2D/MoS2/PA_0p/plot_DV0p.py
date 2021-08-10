import sys, string
import numpy as np
import matplotlib.pyplot as plt

V_0 = np.loadtxt("plavg_0_a3.plot")
V_p = np.loadtxt("plavg_p_a3.plot")

V_diff = -1*(V_0[:,1] - V_p[:,1])

fig, ax = plt.subplots()

ax.plot(V_0[:,0]*0.52918,V_diff,color = 'k',linewidth = 3.2,label = r"-1(V$_0$ - V$_p$)")
ax.axvline(x = V_0[int(len(V_0)*0.241863),0]*0.52918,color = 'darkorange',linestyle = '--',lw = 3.2,label = "Vacancy pos.")
ax.legend(loc = 0,fontsize = 18)
ax.set_xlim(V_0[0,0]*0.52918,V_0[-1,0]*0.52918)
ax.set_ylabel("Potential (eV)",fontsize = 25)
ax.set_xlabel(r"$a_1$ ($\AA$)",fontsize = 25)
ax.tick_params(labelsize = 19, width = 2.5,length = 7)
plt.tight_layout()
plt.savefig("Fig_V_0p.png")
print("Value far from the defect: %2.3f" %( V_diff[int(len(V_0)*0.75)]))
plt.show()
