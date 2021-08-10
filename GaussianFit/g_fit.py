#!/usr/bin/env python
import sys, string
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def pl_avg_a3(vol,a1_dim,a2_dim,a3_dim,step_l,factor):
  A3 = []
  vol_a3 = np.zeros((a3_dim))
  for k in range(a3_dim):
    Sum1 = 0.
    for i in range(a1_dim):
      for j in range(a2_dim):
        Sum1 = Sum1 + vol[i][j][k]
    vol_a3[k] = Sum1/(a2_dim*a1_dim)
    A3.append(k*step_l)
  if factor == "Ryd":
    vol_a3 = vol_a3*rydberg
  elif factor == "Hartree":
    vol_a3 = vol_a3*hartree
  return vol_a3, np.array(A3)

def pl_avg_a1(vol,a1_dim,a2_dim,a3_dim,step_l,factor):
  A1 = []
  vol_a1 = np.zeros((a1_dim))
  for i in range(a1_dim):
    Sum1 = 0.
    for j in range(a2_dim):
      for k in range(a3_dim):
        Sum1 = Sum1 + vol[i][j][k]
    vol_a1[i] = Sum1/(a2_dim*a3_dim)
    A1.append(i*step_l)
  if factor == "Ryd":
    vol_a1 = vol_a2*rydberg
  elif factor == "Hartree":
    vol_a1 = vol_a2*hartree
  return vol_a1,np.array(A1)

def pl_avg_a2(vol,a1_dim,a2_dim,a3_dim,step_l,factor):
  A2 = []
  vol_a2 = np.zeros((a2_dim))
  for j in range(a2_dim):
    Sum1 = 0.
    for i in range(a1_dim):
      for k in range(a3_dim):
        Sum1 = Sum1 + vol[i][j][k]
    vol_a2[j] = Sum1/(a1_dim*a3_dim)
    A2.append(j*step_l)
  if factor == "Ryd":
    vol_a2 = vol_a2*rydberg
  elif factor == "Hartree":
    vol_a2 = vol_a2*hartree
  return vol_a2,np.array(A2)

def read_input(file_name):
  fp = open(file_name,'r')
  lines = fp.readlines()
  plt_dir = 'a1'
  for il in range(len(lines)):
    if "file_name" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_inp = w[1].split()[0]
    if "file_type" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_type = w[1].split()[0]
    if "plt_dir" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      plt_dir = w[1].split()[0]
  return file_inp, file_type,plt_dir

def xsf_read(file):
  fp = open(file,"r")
  lines = fp.readlines()
  primvec = []
  primcoord = []
  grid = []
  vol = []
  origin = []
  for i in range(len(lines)):
    if "PRIMVEC" in lines[i]:
      for j in range(3):
        w = lines[i+j+1].split()
        primvec.append([eval(w[0]),eval(w[1]),eval(w[2])])
    if "PRIMCOORD" in lines[i]:
      w = lines[i+1].split()
      na = eval(w[0])
      for j in range(na):
        w = lines[i+j+2].split()
        primcoord.append([w[0],eval(w[1]),eval(w[2]),eval(w[3])])
    if "DATAGRID_3D_" in lines[i]:
      w = lines[i+1].split()
      grid = [eval(w[0]),eval(w[1]),eval(w[2])]
      w = lines[i+2].split()
      origin = [eval(w[0]),eval(w[1]),eval(w[2])]
      # Skip the next 3 lines
      a1_index = 0
      a2_index = 0
      z_index = 0
      vol = np.zeros((grid[0],grid[1],grid[2]))
      for j in range(6,len(lines)):
        if "END_DATAGRID" in lines[i+j]:
          break
        words = lines[i+j].split()
        words = list(filter(bool, words))
        for w in words:
          vol[a1_index][a2_index][z_index] = eval(w)
          a1_index = a1_index + 1
          if a1_index == grid[0]:
            a2_index = a2_index+1
            a1_index = 0
            if a2_index == grid[1]:
              z_index = z_index + 1
              a2_index = 0
  primvec = np.array(primvec)
  step = np.array([primvec[0]/grid[0],primvec[1]/grid[1],primvec[2]/grid[2]])
  return na, primcoord, grid, origin, step, vol

def cub_read(file):
  ierr = 0
  na = 0
  aspecies = []
  acharge = []
  aposition = []
  grid = []
  origin = []
  step = []
  vol = [[[]]]
  try:
     h = open(file, 'r')
  except:
     ierr = 1
  if ierr == 0:
     for i in range(2):
        s = h.readline()
     s = h.readline()
     t = s.split()
     na = int(t[0])
     origin = []
     for j in range(3):
        origin.append(float(t[j + 1]))
     grid = []
     step = []
     for i in range(3):
        s = h.readline()
        t = s.split()
        grid.append(int(t[0]))
        step.append([])
        for j in range(3):
           step[i].append(float(t[j + 1]))
     for i in range(na):
        s = h.readline()
        t = s.split()
        aspecies.append(int(t[0]))
        acharge.append(float(t[1]))
        aposition.append([])
        for j in range(3):
           aposition[i].append(float(t[j + 2]))
     n = grid[0] * grid[1] * (  int((grid[2] - 1) / 6) + 1)
     i = 0
     j = 0
     k = 0
     for m in range(n):
        s = h.readline()
        t = s.split()
        for l in range(6):
           if k < grid[2]:
              vol[i][j].append(float(t[l]))
              k += 1
        if k == grid[2]:
           k = 0
           j += 1
           if j < grid[1]:
              vol[i].append([])
           else:
              k = 0
              j = 0
              i += 1
              if i < grid[0]:
                 vol.append([[]])
     h.close()
  return ierr, na, aspecies, acharge, aposition, grid, origin, step, vol


def gaussian(xyz,c_x,c_y,c_z,sigma):
  x, y, z = xyz
  g = np.exp(-( (x-c_x)**2 + (y-c_y)**2 + (z-c_z)**2 )/2./sigma**2 )/sigma**3/(2*np.pi)**1.5
  return g#.ravel()

def func1D(x,c_x,sigma):
  return np.exp(-( (x-c_x)**2 )/2./sigma**2 )/sigma**3/(2*np.pi)**1.5





# Read input file
if len(sys.argv) == 1:
  print("Please provide input file after call: plavg.py input_file")
  sys.exit()
file_inp,file_type, dir = read_input(sys.argv[1])
if file_type != "cube" and file_type != "xsf" and file_type != "python":
  print("Please specify file_type in the input file. It takes cube/xsf/python")
  sys.exit()
if file_type == "cube":
  ierr, na, aspecies, acharge, aposition, grid, origin, step, vol = cub_read(file_inp)
elif file_type == "xsf":
  na, primcoord, grid, origin, step, vol = xsf_read(file_inp)

vol = np.array(vol)


fig, ax = plt.subplots()
# Planar average the defect wavefunction charge density
# and plot
if dir == "a1":
  step_l = np.sqrt(np.dot(step[0],step[0]))
  vol_a1, A1 = pl_avg_a1(vol,grid[0],grid[1],grid[2], step_l,None)
  plt.plot(A1,vol_a1, lw = 2.8, label = "wfn")
if dir == "a2":
  step_l = np.sqrt(np.dot(step[1],step[1]))
  vol_a2, A2 = pl_avg_a2(vol,grid[0],grid[1],grid[2], step_l,None)
  plt.plot(A2,vol_a2, lw = 2.8, label = "wfn")
if dir == "a3":
  step_l = np.sqrt(np.dot(step[2],step[2]))
  vol_a3, A3 = pl_avg_a3(vol,grid[0],grid[1],grid[2], step_l,None)
  plt.plot(A3,vol_a3,lw = 2.8,label = "wfn")

print("Shape of the data in the file:", np.shape(vol))
# Flatten array
vol_r = vol.ravel()
celldim1 = np.sqrt(np.dot(step[0],step[0]))*grid[0]
celldim2 = np.sqrt(np.dot(step[1],step[1]))*grid[1]
celldim3 = np.sqrt(np.dot(step[2],step[2]))*grid[2]
x = np.linspace(0, celldim1, grid[0])
y = np.linspace(0, celldim2, grid[1])
z =  np.linspace(0, celldim3, grid[2])
xx, yy, zz = np.meshgrid(x, y, z)
print(np.shape(xx), np.shape(yy))
xyz = np.vstack((xx.ravel(), yy.ravel(), zz.ravel()))
# Fit a Gaussian
popt, pcov = curve_fit(gaussian, xyz,vol_r)
print("Sigma fit: %f bohr"%(popt[3]))
vol_m = gaussian((xx,yy,zz),*popt).reshape(grid[0],grid[1],grid[2])

# Planar average the Gaussian fit and plot
if dir == "a1":
  step_l = np.sqrt(np.dot(step[0],step[0]))
  vol_a1, A1 = pl_avg_a1(vol_m,grid[0],grid[1],grid[2], step_l,None)
  plt.plot(A1,vol_a1,lw = 2.8, label = "Fit",linestyle = '--',color = 'r')
if dir == "a2":
  step_l = np.sqrt(np.dot(step[1],step[1]))
  vol_a2, A2 = pl_avg_a2(vol_m,grid[0],grid[1],grid[2], step_l,None)
  plt.plot(A2,vol_a2,lw = 2.8,label = "Fit",linestyle = '--',color = 'r')
if dir == "a3":
  step_l = np.sqrt(np.dot(step[2],step[2]))
  vol_a3, A3 = pl_avg_a3(vol_m,grid[0],grid[1],grid[2], step_l,None)
  plt.plot(A3,vol_a3,lw = 2.8,label = "Fit",linestyle = '--',color = 'r')
ax.tick_params(labelsize=16,width=2.5,length=6)
plt.legend(fontsize = 16)
plt.tight_layout()
plt.show()
