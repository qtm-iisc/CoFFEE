import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t
@cython.boundscheck(False)
@cython.cdivision(True)

def PS_3Dg(
          double[:,:] B,
          complex[:,:,:] rho_G,
          double eps1_a1,
          double eps1_a2,
          double eps1_a3,
          int imax,
          int jmax,
          int kmax):
  cdef int i,j,k
  cdef double modG,pi,G1,G2,G3
  cdef complex[:,:,:] V_G = np.zeros((2*imax+1, 2*jmax+1, 2*kmax+1),dtype=complex)
  pi = np.pi
  for i in range(-imax, imax+1):
    for j in range(-jmax,jmax+1):
      for k in range(-kmax,kmax+1):
        if i==0 and j==0 and k==0:
          V_G[i+imax,j+jmax,k+kmax] = 0.0
        else:
          G1 = i*B[0,0] + j*B[1,0] + k*B[2,0]
          G2 = i*B[0,1] + j*B[1,1] + k*B[2,1]
          G3 = i*B[0,2] + j*B[1,2] + k*B[2,2]
          modG = eps1_a1*G1**2 + eps1_a2*G2**2 + eps1_a3*G3**2
          V_G[i+imax,j+jmax,k+kmax] = (
                                       4.0*
                                       pi*
                                       rho_G[i+imax,j+jmax,k+kmax]
                                       /
                                       (modG)
                                      )
  return np.array(V_G)
