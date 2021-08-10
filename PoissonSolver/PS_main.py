from .classes import matvec2D,GlobalValues,IFFTnew,matvec1D_py
from scipy.sparse.linalg import LinearOperator, cg, cgs, bicgstab
import numpy as np
import time
#import from matvec1D.so
from .matvec1D import matvec1D
#import from matvec2D.so
from .matvec2D import matvec2D
#import from ps3d.so
from .ps3d import PS_3Dg

# Matrix tomes vector for 2D
def mv(V):
  b_n = matvec2D( GlobalValues.i_curr, GlobalValues.j_curr, GlobalValues.kmax, GlobalValues.c_g.celldm3, np.pi, GlobalValues.eps_g.epsGz_a3,GlobalValues.eps_g.epsGz_a1, GlobalValues.c_g.B[0], GlobalValues.c_g.B[1], V)
  return b_n

# Matrix tomes vector for 1D
def mv_1D(V):
  matvec = matvec1D(V,
                    GlobalValues.lmax,
                    GlobalValues.mmax,
                    GlobalValues.eps_g.eps_GxGy_a1,
                    GlobalValues.eps_g.eps_GxGy_a2,
                    GlobalValues.eps_g.eps_GxGy_a3,
                    GlobalValues.c_g.celldm1,
                    GlobalValues.c_g.celldm2,
                    GlobalValues.c_g.celldm3,
                    GlobalValues.k_curr)
  return matvec
  

def PS_1D(cell_s,charge_s,eps_s,imax,jmax,kmax,comm):
  L = LinearOperator( ( (2*imax+1)*(2*jmax+1), (2*imax+1)*(2*jmax+1)), matvec=mv_1D, dtype=complex)
  cgs_perprocess = int((2*kmax+1)/comm.size)
  Krange = list(range(-kmax, kmax+1))
  tmp = 0
  K_rank = []
  for iproc in range(comm.size):
    K_rank.append(Krange[iproc:len(Krange):comm.size])

  # Construct the charge density and distribute over processes
  if comm.rank == 0:
    charge_s.construct_rho(cell_s,imax,jmax,kmax)
    np.save("rho_r",charge_s.rho_r)
    charge_s.FFT(imax,jmax,kmax)
    for iproc in range(1,comm.size):
      rho_G_rank = np.ndarray(shape = (2*imax+1,2*jmax+1,len(K_rank[iproc])),dtype=complex)
      kk = 0
      for k in K_rank[iproc]:
        rho_G_rank[:,:,kk] = charge_s.rho_G[:,:,k+kmax]
        kk = kk + 1
      comm.Send(rho_G_rank,dest=iproc,tag=iproc)
    rho_G_rank = np.ndarray(shape = (2*imax+1,2*jmax+1,len(K_rank[0])),dtype=complex)
    kk = 0
    for k in K_rank[0]:
      rho_G_rank[:,:,kk] = charge_s.rho_G[:,:,k+kmax]
      kk = kk + 1
  else:
    rho_G_rank = np.ndarray(shape = (2*imax+1,2*jmax+1,len(K_rank[comm.rank])),dtype=complex)
    comm.Recv(rho_G_rank,source=0,tag=comm.rank)
  
  V_G_rank = np.ndarray(shape=(2*imax+1,2*jmax+1,len(K_rank[comm.rank])),dtype=complex)
  V_G = np.ndarray(shape=(2*imax+1,2*jmax+1,2*kmax+1),dtype=complex)

  kk=0
  cg_start = time.time()

  # Start solving the linear equation for every k belonging to this rank
  for k in K_rank[comm.rank]:
    rho_G_k = np.reshape(rho_G_rank[:,:,kk],(np.shape(rho_G_rank[:,:,kk])[0]*np.shape(rho_G_rank[:,:,kk])[1]))
    if k==0:
      bv = (4*np.pi*rho_G_k).tolist()
      # Setting average potential to zero, V(G=0) = 0
      bv[int((len(rho_G_k)-1)/2)] = 0.000000
      L2 = LinearOperator(( (2*imax+1)*(2*jmax+1), (2*imax+1)*(2*jmax+1)), matvec=mv_1D, dtype=complex)
      GlobalValues.k_curr = k
      Soln, info = bicgstab(L2, bv, bv, 0.000001, 1000)
      if info !=0:
        print("error: info, k, iproc", info,k,comm.rank)
      else:
        Soln = Soln.tolist()
        Soln = np.reshape(Soln,(2*imax+1,2*jmax+1))
    else:
      bv = (4*np.pi*rho_G_k).tolist()
      GlobalValues.k_curr = k
      Soln, info = bicgstab(L, bv, bv, 1.e-5, 1000)
      if info !=0:
        print("error: info, k, iproc", info,k,comm.rank)
      else:
        Soln = Soln.tolist()
        Soln = np.reshape(Soln,(2*imax+1,2*jmax+1))
    V_G_rank[:,:,kk] = Soln
    kk = kk + 1
  cg_end = time.time()
  # V_G constructed on every rank. 

  # Now collect them and inverse Fourier transform to get V_r
  V_r = [] 
  if comm.rank != 0:
    comm.Send(V_G_rank,dest=0,tag=comm.rank)
  else:
    kk=0
    for k in K_rank[0]:
      V_G[:,:,k+kmax] = V_G_rank[:,:,kk]
      kk = kk + 1
    for src in range(1,comm.size):
      V_G_tmp = np.ndarray(shape=(2*imax+1,2*jmax+1,len(K_rank[src])),dtype=complex)
      comm.Recv(V_G_tmp,source=src,tag=src)
      kk=0
      for k in K_rank[src]:
        V_G[:,:,k+kmax] = V_G_tmp[:,:,kk]
        kk = kk + 1
    V_r = IFFTnew( V_G, imax, jmax, kmax)
  return V_r 



def PS_3D_new(cell_s,charge_s,eps_s,imax,jmax,kmax,comm):
  # Construct the model charge density in real space
  charge_s.construct_rho(cell_s,imax,jmax,kmax)
  # Fourier transform the model charge density
  charge_s.FFT(imax,jmax,kmax)
  np.save("rho_r",charge_s.rho_r)
  V_G = PS_3Dg(
          cell_s.B,
          charge_s.rho_G,
          eps_s.eps1_a1,
          eps_s.eps1_a2,
          eps_s.eps1_a3,
          imax,
          jmax,
          kmax)
  V_r = IFFTnew( V_G, imax, jmax, kmax)
  return V_r

# This is the python (not cythonized) version of solving
# the 3D Poisson equation. This is not used by default.
def PS_3D(cell_s,charge_s,eps_s,imax,jmax,kmax,comm):
  charge_s.construct_rho(cell_s,imax,jmax,kmax)
  charge_s.FFT(imax,jmax,kmax)
  np.save("rho_r",charge_s.rho_r)
  V_G = np.ndarray( shape = (2*imax+1, 2*jmax+1, 2*kmax+1), dtype = complex)
  for i in range(-imax, imax+1):
    for j in range(-jmax,jmax+1):
      for k in range(-kmax,kmax+1):
        if i==0 and j==0 and k==0:
          V_G[i+imax][j+jmax][k+kmax] = 0.0
        else:
          G1 = i*cell_s.B[0][0] + j*cell_s.B[1][0] + k*cell_s.B[2][0]
          G2 = i*cell_s.B[0][1] + j*cell_s.B[1][1] + k*cell_s.B[2][1]
          G3 = i*cell_s.B[0][2] + j*cell_s.B[1][2] + k*cell_s.B[2][2]
          modG = G1**2 + G2**2 + G3**2
          V_G[i+imax][j+jmax][k+kmax] = 4*np.pi*charge_s.rho_G[i+imax][j+jmax][k+kmax]/(modG*eps_s.eps1_a1)
  V_r = IFFTnew( V_G, imax, jmax, kmax)
  return V_r


def PS_2D(cell_s,charge_s,eps_s,imax,jmax,kmax,comm):
  L = LinearOperator( (2*kmax+1, 2*kmax+1), matvec=mv, dtype=complex)
  cgs_perprocess = int((2*imax+1)/comm.size)
  Irange = list(range(-imax, imax+1))
  tmp = 0
  I_rank = []
  # Distribute the CGs to be solved across the processors using 
  # I_rank
  for i in range(comm.size):
    I_rank.append(Irange[i:len(Irange):comm.size])
  # Construct model charge, fourier transform and distribute among processors.
  if comm.rank == 0:
    charge_s.construct_rho(cell_s,imax,jmax,kmax)
    np.save("rho_r",charge_s.rho_r)
    charge_s.FFT(imax,jmax,kmax)
    for i in range(1, comm.size):
      rho_G_rank = np.ndarray( shape = (len(I_rank[i]), 2*jmax+1, 2*kmax+1), dtype = complex)
      jj = 0
      for j in I_rank[i]:
        rho_G_rank[jj] = charge_s.rho_G[j+imax]
        jj=jj+1
      comm.send(rho_G_rank, dest = i, tag = i)
    rho_G_rank = np.ndarray( shape = (len(I_rank[0]), 2*jmax+1, 2*kmax+1), dtype = complex)
    jj = 0
    for j in I_rank[0]:
      rho_G_rank[jj] = charge_s.rho_G[j+imax]
      jj+=1
  else:
    rho_G_rank = np.ndarray( shape = (len(I_rank[comm.rank]), 2*jmax+1, 2*kmax+1), dtype = complex)
    rho_G_rank = comm.recv(source = 0, tag = comm.rank)
  
  V_G = np.ndarray( shape = (2*imax+1, 2*jmax+1, 2*kmax+1), dtype = complex)
  
  
  temp2 = []
  cg_time=0
  ii = 0
  # Solve a linear equation for every i, j (for every G_1, G_2)
  for i in I_rank[comm.rank]:
    temp1 = []
    for j in range(-jmax, jmax+1):
      if i==0 and j==0:
        bv = (4*np.pi*rho_G_rank[ii][j+jmax][:]).tolist()
        # Setting the average potential to zero: V(G=0) = 0
        bv[kmax] = 0.00000000
        L2 = LinearOperator( (2*kmax+1, 2*kmax+1), matvec=mv, dtype=complex)
        GlobalValues.i_curr = i
        GlobalValues.j_curr = j
        Soln, info = bicgstab(L2, bv, bv, 1.e-5, 8000)
        if info != 0:
          print("Error at", i, j)
        else:
          Soln=Soln.tolist()
          temp1.append(Soln)
      else:
        bv = (4*np.pi*rho_G_rank[ii][j+jmax][:]).tolist()
        start_time_cg = time.time()
        GlobalValues.i_curr = i
        GlobalValues.j_curr = j
        Soln, info = bicgstab(L, bv, bv, 1.e-5, 8000)
        cg_time = cg_time + time.time()-start_time_cg
        if info != 0:
          print("Error at", i, j)
        else:
          Soln=Soln.tolist()
          temp1.append(Soln)
  
    temp2.append(temp1)
    ii+=1
  
  temp2 = np.array(temp2)
  temp1 = np.array(temp1)
  V_r = [] 
  # Collect V_G from all ranks and perform IFFT to get V_r
  V_G_temp = np.ndarray( shape = (2*imax+1, 2*jmax+1, 2*kmax+1), dtype = complex)
  if comm.rank != 0:
    comm.Send(temp2, dest = 0, tag = comm.rank)
  else:
    j=0
    for i in I_rank[0]:
      V_G[i+imax]=temp2[j]
      j=j+1
    for src in range(1,comm.size):
      comm.Recv(V_G_temp[0:len(I_rank[src])], source = src, tag = src)
      j=0
      for i in I_rank[src]:
        V_G[i+imax] = V_G_temp[j]
        j=j+1
    V_r = IFFTnew( V_G, imax, jmax, kmax)
  return V_r
