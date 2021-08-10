#!/usr/bin/env python
##
##   dV_0p.py plots (V_0 - V_p), planar averaged.
##   V_0 is the total DFT potential of the neutral defect
##   V_p is the pristine DFT potential of the same size cell
##   
import sys, string
import numpy as np
from Utilities import *



def read_input(file_name):
  cell_dim = 1.0
  factor = None
  charge = -1
  plt_dir = 'a1'
  fp = open(file_name,'r')
  lines = fp.readlines()
  for il in range(len(lines)):
    if "file_neutral" in lines[il]:
      w = string.split(lines[il].rstrip(),"=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_n = string.split(w[1])[0]
    if "file_pristine" in lines[il]:
      w = string.split(lines[il].rstrip(),"=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_p = string.split(w[1])[0]
    if "file_type" in lines[il]:
      w = string.split(lines[il].rstrip(),"=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_type = string.split(w[1])[0]
    if "plt_dir" in lines[il]:
      w = string.split(lines[il].rstrip(),"=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      plt_dir = string.split(w[1])[0]
    if "factor" in lines[il]:
      w = string.split(lines[il].rstrip(),"=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      factor = w[1].split()[0]
    if "charge" in lines[il]:
      w = string.split(lines[il].rstrip(),"=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      charge = eval(w[1])
  return file_n,file_p,file_type,plt_dir,factor,charge

if len(sys.argv) == 1:
  print("Please provide input file after call: dV_0p.py input_file")
  sys.exit()
# Read the input file
file_n,file_p,file_type,dir,factor,charge = read_input(sys.argv[1])
if dir != "a1" and dir != "a2" and dir != "a3":
  print("Please specify plt_dir in the input file. It takes a1/a2/a3")
  sys.exit()
if file_type != "cube" and file_type != "xsf":
  print("Please specify file_type in the input file. It takes cube/xsf")

# Read the DFT potential files
if file_type == "cube":
  ierr, na, aspecies, acharge, aposition, grid, origin, step, vol_n = cub_read(file_n)
  ierr, na, aspecies, acharge, aposition, grid, origin, step, vol_p = cub_read(file_p)
elif file_type == "xsf":
  na, primcoord, grid, origin, step, vol_n = xsf_read(file_n)
  na, primcoord, grid, origin, step, vol_p = xsf_read(file_p)

# Compute the difference
vol = (np.array(vol_n) - np.array(vol_p))*charge

print("Shape of the data in the file:", np.shape(vol))
if dir == "a1":
  step_l = np.sqrt(np.dot(step[0],step[0]))
  vol_a1, A1 = pl_avg_a1(vol,grid[0],grid[1],grid[2], step_l,factor)
  write2file("pa_dv0p_a1.plot",A1,vol_a1)
if dir == "a2":
  step_l = np.sqrt(np.dot(step[1],step[1]))
  vol_a2, A2 = pl_avg_a2(vol,grid[0],grid[1],grid[2], step_l,factor)
  write2file("pa_dv0p_a2.plot",A2,vol_a2)
if dir == "a3":
  step_l = np.sqrt(np.dot(step[2],step[2]))
  vol_a3, A3 = pl_avg_a3(vol,grid[0],grid[1],grid[2], step_l,factor)
  write2file("pa_dv0p_a3.plot",A3,vol_a3)
