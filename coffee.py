#!/usr/bin/env python
import sys, string
import numpy as np
from math import *
from  PoissonSolver import *
from scipy.sparse.linalg import LinearOperator, cg, cgs, bicgstab
from mpi4py import MPI
import time

ryd = 13.605698066
start_time = time.time()
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

if len(sys.argv) == 1:
  print("Please provide input file after call: coffee.py input_file")
  sys.exit()
#print "Reading input from"
if rank == 0:
  display_init()
  print("Running on %d processor(s)"%(comm.size))

# Initalize the cell class
c = cell()
# Read in the cell parameters from input file, file_in. 
file_in = sys.argv[1]
c.read_params(file_in)
# Display params:
if rank == 0:
  c.disp_params()

# Initalize the dielectric profile class
eps = diel_profile()
# Read in the epsilon profile parameters from input file, file_in. 
eps.read_params(file_in)

# Initalize the gaussian model charge class
charge = gaussian()
# Read in the gaussian paramters from input file, file_in.
if rank == 0:
  charge.read_params(file_in)
  charge.disp_params()

# Initialize the calculation. Compute the FFT and real space grids.
imax, jmax, kmax = c.init_calc()

construct_eps(c,eps,imax,jmax,kmax)
# Display params:
if rank == 0:
  eps.disp_params()

if rank==0:
  print("Grid: %d, %d, %d"%( 2*imax + 1, 2*jmax + 1, 2*kmax + 1))

GlobalValues.c_g = c
GlobalValues.kmax = kmax
GlobalValues.lmax = imax
GlobalValues.mmax = jmax
GlobalValues.eps_g = eps

V_r = Solver(c,charge,eps,imax,jmax,kmax,comm)
if rank==0:
#  V_r = IFFTnew( V_G, imax, jmax, kmax)
  ComputeEnergy(V_r,charge.rho_r,imax,jmax,kmax,c)
  np.save("V_r",V_r*2.*ryd)
Exec_time=time.time()-start_time
if rank == 0:
  print("Execution time: %.2f s"%(Exec_time))
