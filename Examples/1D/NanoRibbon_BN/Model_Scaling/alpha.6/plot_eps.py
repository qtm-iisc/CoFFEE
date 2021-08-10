#!/usr/bin/python
import sys, string
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.path as mplPath
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

celldm1 = 28.345742469481085*0.52918
celldm2 = 65.6793667152*0.52918

eps_xy_a1 = np.load("eps_xy_a1.npy")
eps_xy_a2 = np.load("eps_xy_a2.npy")
eps_xy_a3 = np.load("eps_xy_a3.npy")
a1_dim, a2_dim = np.shape(eps_xy_a1)[0], np.shape(eps_xy_a1)[1]
fig, axarr = plt.subplots(1,3,figsize = (18,5))
map_c = 'summer'
a1 = np.arange(0,celldm1,celldm1/a1_dim)
a2 = np.arange(0,celldm2,celldm2/a2_dim)
h_map = axarr[0].pcolor(a2,a1,eps_xy_a1,cmap=map_c)
cb = fig.colorbar(h_map,shrink=0.9,ax = axarr[0])
cb.ax.tick_params(labelsize=22)
h_map = axarr[1].pcolor(a2,a1,eps_xy_a2,cmap=map_c)
cb = fig.colorbar(h_map,shrink=0.9,ax = axarr[1])
cb.ax.tick_params(labelsize=20)
h_map = axarr[2].pcolor(a2,a1,eps_xy_a3,cmap=map_c)
cb = fig.colorbar(h_map,shrink=0.9,ax = axarr[2])
cb.ax.tick_params(labelsize=22)

for i in range(3):
  axarr[i].set_xlabel(r"$y (\AA)$",fontsize = 30)
  axarr[i].set_ylabel(r"$x (\AA)$",fontsize = 30)
  axarr[i].set_ylim(0,celldm1-celldm1/a1_dim)
  axarr[i].set_xlim(0,celldm2-celldm2/a2_dim)
  axarr[i].tick_params(labelsize=22,width=1.8)

axarr[0].set_title(r"Dielectric profile along a$_1$/x",fontsize = 22)
axarr[1].set_title(r"Dielectric profile along a$_2$/y",fontsize = 22)
axarr[2].set_title(r"Dielectric profile along a$_3$/z",fontsize = 22)
plt.tight_layout()
plt.savefig("Ribbon_profile.png",dpi = 400)
plt.show()
