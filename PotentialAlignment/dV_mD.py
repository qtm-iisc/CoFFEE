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
  fp = open(file_name,'r')
  lines = fp.readlines()
  for il in range(len(lines)):
    if "file_charged" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_q = w[1].split()[0]
    if "file_neutral" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_n = w[1].split()[0]
    if "file_model" in lines[il]:
      w = lines[il].split("=")
      if len(w) < 2 or len(w) > 3:
        print("ERROR while parsing input file: %s, line: %d"%(file_name,il))
        sys.exit()
      file_m_py = w[1].split()[0]
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
  return file_q,file_n,file_m_py,file_type,plt_dir,factor

if __name__ == "__main__":

  # Check for the input file
  if len(sys.argv) == 1:
    print("Please provide input file after call: dV_mD.py input_file")
    sys.exit()
  
  # Read the input file
  file_q,file_n,file_m_py,file_type,dir,factor = read_input(sys.argv[1])
  if dir != "a1" and dir != "a2" and dir != "a3":
    print("Please specify plt_dir in the input file. It takes a1/a2/a3")
    sys.exit()
  if file_type != "cube" and file_type != "xsf":
    print("Please specify file_type in the input file. It takes cube/xsf")
    sys.exit()

  # Read the DFT potential files
  if file_type == "cube":
    ierr, na, aspecies, acharge, aposition, grid, origin, step, vol_q = cub_read(file_q)
    ierr, na, aspecies, acharge, aposition, grid, origin, step, vol_n = cub_read(file_n)
  elif file_type == "xsf":
    na, primcoord, grid, origin, step, vol_q = xsf_read(file_q)
    na, primcoord, grid, origin, step, vol_n = xsf_read(file_n)


  # Read the model potential (python) file:
  grid_py,vol_m = py_read(file_m_py)

  # We will use the same cell_dim for model potential plot 
  # as read from the DFT potential file.
  

  # DFT difference potential:
  vol = np.array(vol_q) - np.array(vol_n)
  print "dir", dir

  print("Shape of the data in the file:", np.shape(vol))
  print("The first column in the files is in bohr")
  
  # Planar average the DFT difference and the model potential
  # and write the plot to file
  if dir == "a1":
    step_l = np.sqrt(np.dot(step[0],step[0]))
    vol_a1, A1 = pl_avg_a1(vol,grid[0],grid[1],grid[2], step_l,factor)
    write2file("DFTdiff_a1.plot",A1,vol_a1)
    # cell_dim1 is step_l*grid[0], in bohr
    step_n = step_l*grid[0]/grid_py[0]
    vol_a1_py, A1_py = pl_avg_a1(vol_m,grid_py[0],grid_py[1],grid_py[2], step_n,None)
    write2file("model_a1.plot",A1_py,vol_a1_py)
  if dir == "a2":
    step_l = np.sqrt(np.dot(step[1],step[1]))
    vol_a2, A2 = pl_avg_a2(vol,grid[0],grid[1],grid[2], step_l,factor)
    write2file("DFTdiff_a2.plot",A2,vol_a2)
    # cell_dim2 is step_l*grid[1], in bohr
    step_n = step_l*grid[1]/grid_py[1]
    vol_a2_py, A2_py = pl_avg_a2(vol_m,grid_py[0],grid_py[1],grid_py[2], step_n,None)
    write2file("model_a2.plot",A2_py,vol_a2_py)
  if dir == "a3":
    step_l = np.sqrt(np.dot(step[2],step[2]))
    vol_a3, A3 = pl_avg_a3(vol,grid[0],grid[1],grid[2], step_l,factor)
    write2file("DFTdiff_a3.plot",A3,vol_a3)
    # cell_dim3 is step_l*grid[2], in bohr
    step_n = step_l*grid[2]/grid_py[2]
    vol_a3_py, A3_py = pl_avg_a3(vol_m,grid_py[0],grid_py[1],grid_py[2], step_n,None)
    write2file("model_a3.plot",A3_py,vol_a3_py)
  sys.exit()
