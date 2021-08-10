#rom libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
def matvec2D(int i,int j,int kmax, double celldm3, double pi, complex[:] epsGz_perp,complex[:] epsGz_ll, double[:] b_1, double[:] b_2, complex[:] V):
  cdef double Gz1, Gz2, Gx,Gy,pi2_celldm3
  cdef int k_t,k1, k2, j1, p, q, index
  cdef complex[:] b_n = np.zeros(2*kmax+1, dtype = complex)
  cdef complex Sum
  index = 0
  Gx = i*b_1[0] + j*b_2[0]
  Gy = i*b_1[1] + j*b_2[1]
  pi2_celldm3= 2.*pi/celldm3
  for k1 in range(-kmax,kmax+1):
    Gz1 = k1*pi2_celldm3
    Sum = 0.
    j1 = 0
    for k2 in range(-kmax,kmax+1):
      Gz2 = k2*pi2_celldm3
      Sum+= (epsGz_perp[2*kmax+(k1-k2)]*Gz2*Gz1 + epsGz_ll[2*kmax+(k1-k2)]*( Gx**2 + Gy**2) )*V[j1]
      j1 = j1+1
    b_n[index]=Sum
    index+=1
  return b_n
