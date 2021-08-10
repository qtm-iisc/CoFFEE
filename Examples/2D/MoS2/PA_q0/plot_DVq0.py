import sys, string
import numpy as np
import matplotlib.pyplot as plt

V_0 = np.loadtxt("plavg_0_a3.plot")
V_q = np.loadtxt("plavg_q_a3.plot")
V_m = np.loadtxt("../Model_Scaling/alpha.6/Plot/plavg_a3.plot")

V_diff = -1*(V_q[:,1] - V_0[:,1])

fig, ax = plt.subplots()

ax.plot(V_0[:,0]*0.52918,V_diff,color = 'k',linewidth = 4.3,label = "DFT diff.")
ax.plot(V_m[:,0]*0.52918,V_m[:,1],color = 'r',lw = 4.3,linestyle = '--',label = "Model")
ax.axvline(x = V_0[int(len(V_0)*0.241863),0]*0.52918,color = 'darkorange',linestyle = '--',lw = 3.8,label = "Vacancy pos.")
ax.legend(loc = 0,fontsize = 23)
ax.set_xlim(V_0[0,0]*0.52918,V_0[-1,0]*0.52918)
ax.set_ylabel("Potential (eV)",fontsize = 35)
ax.set_xlabel(r"$a_3$ ($\AA$)",fontsize = 35)
ax.set_ylim(-0.6,0.5)
ax.tick_params(labelsize = 23, width = 2.5,length = 7)
plt.tight_layout()
plt.savefig("Fig_V_q0.png")
plt.show()
