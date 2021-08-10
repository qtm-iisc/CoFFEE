import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t
@cython.boundscheck(False)

def matvec1D(
             complex[:] V,
             int lmax,
             int mmax,
             complex[:,:] eps_GxGy_a1,
             complex[:,:] eps_GxGy_a2,
             complex[:,:] eps_GxGy_a3,
             double celldm1,
             double celldm2,
             double celldm3,
             int k):
  
  cdef int l1,m1,j1,l2,m2,im
  cdef double tpbco,tpbct,tpbctwo,Gx1,Gy1,Gx2,Gy2,Gz,t1,t2,t3,pi
  cdef complex Sum
  cdef complex[:] matvec = np.zeros((2*lmax+1)*(2*mmax+1),dtype=complex)
  im = 0
  pi = 3.1415926535897932384626433832795028
  tpbco = 2.*pi/celldm1
  tpbctwo = 2.*pi/celldm2
  tpbct = 2.*pi/celldm3
  for l1 in range(-lmax,lmax+1):
    Gx1 = l1*tpbco
    for m1 in range(-mmax,mmax+1):
      Gy1 = m1*tpbctwo
      Sum = 0.0
      j1 = 0
      for l2 in range(-lmax,lmax+1):
        Gx2 = l2*tpbco
        for m2 in range(-mmax, mmax+1):
          Gy2 = m2*tpbctwo
          Gz = k*tpbct
          Sum += (
                  (Gx2*Gx1)
                  *eps_GxGy_a1[2*lmax+(l1-l2),2*mmax+(m1-m2)]
                  + ( Gy2*Gy1)
                  *eps_GxGy_a2[2*lmax+(l1-l2),2*mmax+(m1-m2)]
                  + Gz*Gz
                  *eps_GxGy_a3[2*lmax+(l1-l2),2*mmax+(m1-m2)]
                  )*V[j1]
          j1 = j1+1
      matvec[im] = Sum
      im = im + 1
  return matvec
