#!/usr/bin/env python

#-----------------
#  plavg.py
#  Computes planar average of 3D data
#  written by Mit Naik (March 2017)
#-----------------
import sys, string
import numpy as np

bohr = 0.52917721092
rydberg = 13.60569253
hartree = 27.21138505
inf9 = 1.0e+9

def read_input(file_name):
  cell_dim = 1.0
  fp = open(file_name,'r')
  lines = fp.readlines()
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
    if "factor" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      factor = w[1].split()[0]
    if "cell_dim" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      cell_dim = eval(w[1])
      
  return file_inp,file_type,plt_dir,factor,cell_dim
      
    

def write2file(file_name,A,v_a):
  fp = open(file_name,'w')
  if len(A) != len(v_a):
    print("Error: len(A) != len(v_a)")
  for i in range(len(A)):
    fp.write("%4.3f %4.8f\n"%(A[i],v_a[i]))
  fp.close()

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
    vol_a1 = vol_a1*rydberg
  elif factor == "Hartree":
    vol_a1 = vol_a1*hartree
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

def py_read(file):
  vol = np.load(file)
  grid = np.shape(vol)
  return grid,vol

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
      n = grid[0] * grid[1] * ((grid[2] - 1) / 6 + 1)
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

if __name__ == "__main__":
  import sys, string
  import math
  import numpy as np
  
  if len(sys.argv) == 1:
    print("Please provide input file after call: plavg.py input_file")
    sys.exit()
  
  file_inp,file_type,dir,factor,cell_dim = read_input(sys.argv[1])
  if dir != "a1" and dir != "a2" and dir != "a3":
    print("Please specify plt_dir in the input file. It takes a1/a2/a3")
    sys.exit()
  if file_type != "cube" and file_type != "xsf" and file_type != "python":
    print("Please specify file_type in the input file. It takes cube/xsf/python")
  if file_type == "cube":
    ierr, na, aspecies, acharge, aposition, grid, origin, step, vol = cub_read(file_inp) 
  elif file_type == "xsf":
    na, primcoord, grid, origin, step, vol = xsf_read(file_inp)
  elif file_type == "python":
    grid,vol = py_read(file_inp)
  
  vol = np.array(vol)
  print("Shape of the data in the file:", np.shape(vol))
  if dir == "a1":
    if file_type == "python":
      step_l = cell_dim/grid[0]
    else:
      step_l = np.sqrt(np.dot(step[0],step[0]))
    vol_a1, A1 = pl_avg_a1(vol,grid[0],grid[1],grid[2], step_l,factor)
    write2file("plavg_a1.plot",A1,vol_a1)
  if dir == "a2":
    if file_type == "python":
      step_l = cell_dim/grid[1]
    else:
      step_l = np.sqrt(np.dot(step[1],step[1]))
    vol_a2, A2 = pl_avg_a2(vol,grid[0],grid[1],grid[2], step_l,factor)
    write2file("plavg_a2.plot",A2,vol_a2)
  if dir == "a3":
    if file_type == "python":
      step_l = cell_dim/grid[2]
    else:
      step_l = np.sqrt(np.dot(step[2],step[2]))
    vol_a3, A3 = pl_avg_a3(vol,grid[0],grid[1],grid[2], step_l,factor)
    write2file("plavg_a3.plot",A3,vol_a3)
  sys.exit()
