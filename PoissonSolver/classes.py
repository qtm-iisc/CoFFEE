import sys, string
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, cgs, bicgstab
from math import *
#from mpi4py import MPI
import matplotlib.path as mplPath
import time

def display_init():
  # Initial display 
  print("########################################################")
  print("CoFFEE: Corrections For Formation Energies and ")
  print("        Eigenvalues for charged defect simulations")
  print("########################################################")

  

def IFFTnew(F_G, lmax, mmax, nmax):
  #
  # Compute 3D inverse Fourier transform of F_G. 
  # Dimension of the IFFT: (2*lmax+1,2*mmax+1,2*nmax+1)
  #
  F_r_tr = np.fft.ifftn(np.fft.ifftshift(F_G*(2*lmax+1)*(2*mmax+1)*(2*nmax+1)))
  return F_r_tr

def ComputeEnergy(V_r,rho_r,lmax,mmax,nmax,cell_s):
  #
  # Calculate and print the electrostatic energy using the potential generated
  # Integral of 0.5*V_r*dV
  # Function takes inputs: real-space potential, V_r, charge density, rho_r
  # Dimension of the arrays: (2*lmax+1,2*mmax+1,2*nmax+1)
  # cell_s: object of the class cell
  #
  Vol = np.dot(cell_s.A2[0], np.cross(cell_s.A2[1],cell_s.A2[2]))
  print("Volume: %4.3f"%( Vol))
  dV = (1./float(2*lmax+1)/float(2*mmax+1)/float(2*nmax+1))*Vol
  Sum = 0.0
  for l in range(2*lmax+1):
    for m in range(2*mmax+1):
      for n in range(2*nmax+1):
        Sum+= 0.5*V_r[l][m][n]*rho_r[l][m][n]*dV
  print("!  Total Energy (eV): %.4f"%( np.real(Sum)*13.60569253*2.))

def matvec2D(V,kmax,eps,c,i,j):
  #
  # Python routine to perform the matvec for the Poisson equation 
  # solved in the case of a 2D system. This routine is not used by 
  # default to compute the matvec. The Cython routine is used by default. 
  # Inputs:
  # V: potential array; the x of Ax=b
  # kmax: dimension of V is 2*kmax+1
  # eps: object of the class epsilon, initialised at the beginning
  # c: object of the class cell, initialised at the beginning
  # i,j: indices of the G1, G2 vectors we are solving the linear equation for.
  #
  a_n = []
  K1 = list(range(-kmax, kmax+1))
  K2 = list(range(-kmax, kmax+1))
  # Remove the G=0 element. We set V(G=0) = 0.
  if i == 0. and j == 0.:
    del K1[kmax]
    del K2[kmax]
  for k1 in K1:
    Gz1 = k1*2.*np.pi/c.celldm3
    Sum = 0.
    j1 = 0
    for k2 in K2:
      Gz2 = k2*2.*np.pi/c.celldm3
      Gx = i*c.B[0][0] + j*c.B[1][0]
      Gy = i*c.B[0][1] + j*c.B[1][1]
      # Eqn 10 in the paper.
      Sum+= (eps.epsGz_a3[2*kmax+(k1-k2)]*Gz2*Gz1 + eps.epsGz_a1[2*kmax+(k1-k2)]*( Gx**2) + \
             eps.epsGz_a2[2*kmax+(k1-k2)]*(Gy**2 ))*V[j1]
      j1 = j1+1
    b_n.append(Sum)
  return np.array(b_n, dtype = complex)

def matvec1D_py(V,lmax,mmax,eps,c,k):
  #
  # Python routine to perform the matvec for the Poisson equation 
  # solved in the case of a 1D system. This routine is not used by 
  # default to compute the matvec. The Cython routine is used by default. 
  # Inputs:
  # V: potential array; the x of Ax=b
  # lmax,mmax: dimension of V is (2*lmax+1,2*mmax+1)
  # eps: object of the class epsilon, initialised at the beginning
  # c: object of the class cell, initialised at the beginning
  # k: index of the G3 vector we are solving the linear equation for.
  #
  #
  matvec = []
  L1 = list(range(-lmax, lmax+1))
  L2 = list(range(-lmax, lmax+1))
  M1 = list(range(-mmax, mmax+1))
  M2 = list(range(-mmax, mmax+1))
  for l1 in L1:
    Gx1 = l1*2.*np.pi/c.celldm1
    for m1 in M1:
      Gy1 = m1*2.*np.pi/c.celldm2
      Sum = 0.0
      j1 = 0
      for l2 in L2:
        Gx2 = l2*2.*np.pi/c.celldm1
        for m2 in M2:
          Gy2 = m2*2.*np.pi/c.celldm2
          Gz = k*2.*np.pi/c.celldm3
          # Eqn 11 in the paper. 
          Sum+= ( (Gx2*Gx1)*eps.eps_GxGy_a1[2*lmax+(l1-l2)][2*mmax+(m1-m2)]
                  (Gy2*Gy1)*eps.eps_GxGy_a2[2*lmax+(l1-l2)][2*mmax+(m1-m2)]
                  + Gz**2*eps.eps_GxGy_a3[2*lmax+(l1-l2)][2*mmax+(m1-m2)] 
                  )*V[j1]
          j1 = j1+1
      matvec.append(Sum)
  matvec = np.array(matvec, dtype = complex)
  return matvec


class cell:
  #
  # The cell class defines the cell parameters for the 
  # model calculation. It also sets up the reciprocal 
  # space grid based on the plane wave energy cut off. 
  # 
  def __init__(self):
    #
    # A stores the lattice vectors
    # B stores the reciprocal lattice vectors
    # celldm(1 to 3) are the cell dimension
    # ecut is the plane wave energy cut-off
    #
    self.A = np.zeros((3,3))
    self.B = np.zeros((3,3))
    self.A2 = np.zeros((3,3))
    self.celldm1 = 0.0
    self.celldm2 = 0.0
    self.celldm3 = 0.0
    self.ecut = 18.0

  def init_calc(self):
    #
    # Initialises the calculation. Sets up the 
    # grid in reciprocal space.
    #
    a_1 = self.A2[0]
    a_2 = self.A2[1]
    a_3 = self.A2[2]

    self.B = 2*np.pi*np.linalg.inv(np.transpose(self.A2))
    b_1 = 2*np.pi*np.cross(a_2,a_3)/np.dot(a_1, np.cross(a_2,a_3))
    b_2 = 2*np.pi*np.cross(a_3,a_1)/np.dot(a_1, np.cross(a_2,a_3))
    b_3 = 2*np.pi*np.cross(a_1,a_2)/np.dot(a_1, np.cross(a_2,a_3))

    Gmax = np.sqrt(2*self.ecut)
    imax = int( Gmax/np.sqrt(np.dot(b_1, b_1)) )+1
    jmax = int( Gmax/np.sqrt(np.dot(b_2, b_2)) )+1
    kmax = int( Gmax/np.sqrt(np.dot(b_3, b_3)) )+1 
    return imax,jmax,kmax
  
  def disp_params(self):
    #
    # Display the simulation cell parameter read from the 
    # input file. 
    #
    print("CELL PARAMETERS:")
    print("Cell dimensions (bohr): %4.3f, %4.3f,%4.3f "%( \
    self.celldm1, self.celldm2, self.celldm3))
    print("Lattice vectors (normalized):")
    print("a1: %4.3f, %4.3f, %4.3f"%( self.A[0][0],self.A[0][1],self.A[0][2]))
    print("a2: %4.3f, %4.3f, %4.3f"%( self.A[1][0],self.A[1][1],self.A[1][2]))
    print("a3: %4.3f, %4.3f, %4.3f"%( self.A[2][0],self.A[2][1],self.A[2][2]))
    print("Plane-wave energy cut-off (Ry): %4.3f \n"%( self.ecut*2))

  def read_params(self,file_name):
    #
    # Read "&CELL_PARAMETERS" section of the input file.
    #
    fp = open(file_name,'r')
    lines = fp.readlines()
    for il in range(len(lines)):
      if "&CELL_PARAMETERS" in lines[il]:
         for jl in range(il+1,len(lines)):
           if "/" in lines[jl]:
             break
           if "Lattice_Vectors" in lines[jl]:
             for ilv in range(3):
               w = lines[ilv+jl+1].split()
               w = list(filter(bool,w))
               self.A[ilv] = [eval(w[0]),eval(w[1]),eval(w[2])]
             self.A = np.array(self.A)
           elif "Cell_dimensions" in lines[jl]:
             w = lines[jl].split()
             w = list(filter(bool,w))
             if len(w) > 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             elif len(w)==2:
               if w[1] == "bohr":
                 bohr_flag = True
               elif w[1] == "angstrom":
                 bohr_flag = False
               else:
                 print("ERROR while parsing input file; wrong units:%s, line: %d"%(file_name,jl+1))
                 sys.exit()
             else:
               bohr_flag = True
             w = lines[jl+1].split()
             w = list(filter(bool,w))
             if len(w) > 3:
               print("ERROR while parsing input file; too much data:%s, line: %d"%(file_name,jl+1+1))
               sys.exit()
             else:
               if bohr_flag:
                 self.celldm1 = eval(w[0])
                 self.celldm2 = eval(w[1])
                 self.celldm3 = eval(w[2])
               else:
                 self.celldm1 = eval(w[0])/0.529177249
                 self.celldm2 = eval(w[1])/0.529177249
                 self.celldm3 = eval(w[2])/0.529177249
           elif "Ecut" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) < 2 or len(w) > 3:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) == 2:
               if w1[1] == "Hartree":
                 self.ecut = eval(w1[0])
               elif w1[1] == "Rydberg":
                 self.ecut = eval(w1[0])*0.5
             else:
               self.ecut = eval(w1[0])
    self.A2 = np.array([self.A[0]*self.celldm1,self.A[1]*self.celldm2,self.A[2]*self.celldm3])
             
class diel_profile:
  def __init__(self):
    #
    # Initialise attributes of the class.
    #
    # Profile for 2D systems, along a1, a2 and a3 directions.
    self.epsZ_a1 = []
    self.epsZ_a2 = []
    self.epsZ_a3 = []
    # FT of the above arrays
    self.epsGz_a1 = []
    self.epsGz_a2 = []
    self.epsGz_a3 = []
    # Sets the profile type: "Slab/Gaussian/Wire/Ribbon"
    self.Profile =  "Slab"
    # eps1_* is the value of epsilon inside the material
    # eps2_* is the value of epsilon outside.
    self.eps1_a1 = 1.0
    self.eps2_a1 = 1.0
    self.eps1_a2 = 1.0
    self.eps2_a2 = 1.0
    self.eps1_a3 = 1.0
    self.eps2_a3 = 1.0
    # Profile for 1D systems, along a1, a2 and a3 directions.
    self.eps_xy_a1 = [] 
    self.eps_xy_a2 = [] 
    self.eps_xy_a3 = []
    # Fourier transform of the above arrays
    self.eps_GxGy_a1 = []
    self.eps_GxGy_a2 = []
    self.eps_GxGy_a3 = []
    # Flag to write epsilon profile to file
    self.plot_eps = True

    # Slab/Gaussian profile properties
    self.width = 1.0
    self.center = 1.0
    self.gauss_amp_a1 = 0.0
    self.gauss_amp_a2 = 0.0
    self.gauss_amp_a3 = 0.0
    self.gauss_sigma = 1.0
    self.smp = 1.0
    # Wire profile properties
    self.vertices_file = ""
    self.circle = False
    self.radius = 0.0
    self.center_a1 = 0.5
    self.center_a2 = 0.0
    self.vertices = []
    # Ribbon profile properties
    self.center_x = 0.0
    self.center_y = 0.0
    self.width_y = 1.0
    self.width_x = 1.0
    self.gauss_along_x = True
    self.gauss_along_y = False


  def disp_params(self):
    #
    # Displays the parameters read from input file under the 
    # &DIELECTRIC PARAMETERS section
    #
    print("DIELECTRIC PARAMETERS")
    if self.Profile == "Bulk":
      print("Profile: Bulk")
      print("epsilon tensor:")
      e_tensor = np.array([[self.eps1_a1,0.0, 0.0],[0.0,self.eps1_a2,0.0], \
                           [0.0,0.0,self.eps1_a3]])
      print(e_tensor)
    elif self.Profile == "Slab":
      print("Profile:", self.Profile)
      print("Epsilon tensor inside the material:")
      e_tensor = np.array([[self.eps1_a1,0.0, 0.0],[0.0,self.eps1_a2,0.0], \
                           [0.0,0.0,self.eps1_a3]])
      print(e_tensor)
      print("Epsilon tensor outside the material:")
      e_tensor = np.array([[self.eps2_a1,0.0, 0.0],[0.0,self.eps2_a2,0.0], \
                           [0.0,0.0,self.eps2_a3]])
      print(e_tensor)
      print("Slab width (bohr):", self.width)
      print("Slab center (bohr):", self.center)
      print("Smoothness parameter (bohr):", self.smp)
    elif self.Profile == "Ribbon":
      print("Profile:", self.Profile)
      print("Epsilon tensor inside the material:")
      e_tensor = np.array([[self.gauss_amp_a1,0.0, 0.0],[0.0,self.gauss_amp_a2,0.0], \
                           [0.0,0.0,self.gauss_amp_a3]])
      print(e_tensor)
      if self.gauss_along_x:
        print("Gaussian profile is along x")
        print("Center of the gaussian (bohr):", self.center_x)
        print("Width of the gaussian (bohr):", self.width_x)
      elif self.gauss_along_y:
        print("Gaussian profile is along y")
        print("Center of the gaussian (bohr):", self.center_y)
        print("Width of the gaussian (bohr):", self.width_y)
      if not self.gauss_along_x:
        print("Slab profile along x")
        print("Center of the slab along x (bohr):", self.center_y)
        print("Width of the slab along x (bohr):", self.width_y)
      if not self.gauss_along_y:
        print("Slab profile along y")
        print("Center of the slab along y (bohr):", self.center_y)
        print("Width of the slab along y (bohr):", self.width_y)
    print("\n")    
        
        
      

  def construct_ribbon(self,gauss_amp,gauss_along_x,gauss_along_y,sigma,
                       c_x,c_y,w_x,w_y,smp,lmax,mmax,cell_s):
    #
    # Constructs the ribbon dielectric profile
    # Constructs a slab-like profile along x and y directions
    # unless gauss_along_x or gauss_along_y are specified
    #
    # Inputs:
    # gauss_amp: Amplitude/max height of the Gaussian
    # gauss_along_x: Flag to construct Gaussian along x
    # gauss_along_y: Flag to construct Gaussian along y
    # sigma: 
    # c_x, c_y: Center of the Gaussian
    # w_x, w_y: Width of the slab-like profile
    # smp: smoothening parameter at the edges of the slab profile
    # lmax, mmax: Dimensions of the profile: (2*lmax+1,2*mmax+1)
    #             Set during initialisation of the calculation
    #             depending on the energy cut-off.
    # cell_s: Object of the class cell, initialised at the beginning 
    #
    # Returns:
    # eps_xy: The ribbon profile
    #
    eps_xy = np.zeros((2*lmax+1,2*mmax+1))

    a1_list = np.arange(0,1,1./(2*lmax+1))
    a2_list = np.arange(0,1,1./(2*mmax+1))
   
    c_crys_x = c_x/cell_s.celldm1
    c_crys_y = c_y/cell_s.celldm2
    
    c_crys_x_n = np.searchsorted(a1_list,c_crys_x)/float(2*lmax+1)
    c_crys_y_n = np.searchsorted(a2_list,c_crys_y)/float(2*mmax+1)

    c_x = c_crys_x_n*cell_s.celldm1
    c_y = c_crys_y_n*cell_s.celldm2
    
    a1_list = a1_list*cell_s.celldm1
    a2_list = a2_list*cell_s.celldm2

    # Facilitate periodic boundary conditions.
    # Checking left and right of the box along a1
    # overflow flags: of_* are set after checks.
    of_a1_l = False
    of_a1_r = False
    of_a2_l = False
    of_a2_r = False
    if gauss_along_x:
      if c_x - 4*sigma < 0.:
        of_a1_l = True
      if c_x + 4*sigma > cell_s.celldm1:
        of_a1_r = True
      if c_y - (w_y/2 + 2*smp) < 0.:
        of_a2_l = True
      if c_y + (w_y/2 + 2*smp) > cell_s.celldm2:
        of_a2_r = True
    else:
      if c_y - 4*sigma < 0.:
        of_a2_l = True
      if c_y + 4*sigma > cell_s.celldm2:
        of_a2_r = True
      if c_x - (w_x/2 + 2*smp) < 0.:
        of_a1_l = True
      if c_x + (w_x/2 + 2*smp) > cell_s.celldm1:
        of_a1_r = True

    # Initially construct Gaussian with gauss_amp - 1; Add 1 at the end
    amp_sqrt = (gauss_amp-1)**0.5 
    l=0
    #
    # Construct the ribbon profile in accordance with the input flags, 
    # respecting the periodic boundary consitions.
    #
    for x_e in a1_list:
      if gauss_along_x:
        if of_a1_l and cell_s.celldm1 - x_e < 4*sigma:
          x = 0 - (cell_s.celldm1 - x_e)
        elif of_a1_r and x_e < 4*sigma:
          x = cell_s.celldm1 + x_e
        else:
          x = x_e
      else:
        if of_a1_l and cell_s.celldm1 - x_e < (w_x/2 + 2*smp):
          x = 0 - (cell_s.celldm1 - x_e)
        elif of_a1_r and x_e < (w_x/2 + 2*smp):
          x = cell_s.celldm1 + x_e
        else:
          x = x_e
      m=0  
      for y_e in a2_list:
        if gauss_along_y:
          if of_a2_l and cell_s.celldm2 - y_e < 4*sigma:
            y = 0 - (cell_s.celldm2 - y_e)
          elif of_a2_r and y_e < 4*sigma:
            y = cell_s.celldm2 + y_e
          else:
            y = y_e
        else:
          if of_a2_l and cell_s.celldm2 - y_e < (w_y/2 + 2*smp):
            y = 0 - (cell_s.celldm2 - y_e)
          elif of_a2_r and y_e < (w_y/2 + 2*smp):
            y = cell_s.celldm2 + y_e
          else:
            y = y_e
        
        if gauss_along_x:
          eps_x = amp_sqrt*np.exp(-1.*(x-c_x)**2/(2*sigma**2))
          eps_y = ((0.5*(-1*amp_sqrt)*erf( (y-(c_y+w_y/2.))/smp)) 
                   - (0.5*(-1*amp_sqrt)*erf( (y-(c_y-w_y/2.))/smp)))
          eps_xy[l][m] = eps_x*eps_y
        else:
          eps_y = amp_sqrt*np.exp(-1.*(y-c_y)**2/(2*sigma**2))
          eps_x = ((0.5*(-1*amp_sqrt)*erf( (x-(c_x+w_x/2.))/smp)) 
                   - (0.5*(-1*amp_sqrt)*erf( (x-(c_x-w_x/2.))/smp)) )
          eps_xy[l][m] = eps_x*eps_y
        m = m + 1
      l = l + 1

    # This ensures the epsilon outside the material (in vacuum) is 1 
    eps_xy = eps_xy + 1
    return eps_xy

  def construct_epsxy(self,cell_s,lmax,mmax):
    # 
    # Constructs Wire or Ribbon profile and updates the 
    # relevant attributes of the epsilon class object calling it. 
    #
    if self.Profile == "Wire":
      self.eps_xy_a3 = self.construct_wire(self.eps1_a3,self.eps2_a3,cell_s,lmax,mmax,self.circle,self.vertices_file)
      self.eps_xy_a2 = self.construct_wire(self.eps1_a2,self.eps2_a2,cell_s,lmax,mmax,self.circle,self.vertices_file)
      self.eps_xy_a1 = self.construct_wire(self.eps1_a1,self.eps2_a1,cell_s,lmax,mmax,self.circle,self.vertices_file)
    if self.Profile == "Ribbon":  
      self.eps_xy_a3 = self.construct_ribbon(self.gauss_amp_a3,self.gauss_along_x,
        self.gauss_along_y,self.gauss_sigma,self.center_x,self.center_y,self.width_x,
        self.width_y,self.smp,lmax,mmax,cell_s)
      self.eps_xy_a2 = self.construct_ribbon(self.gauss_amp_a2,self.gauss_along_x,
        self.gauss_along_y,self.gauss_sigma,self.center_x,self.center_y,self.width_x,
        self.width_y,self.smp,lmax,mmax,cell_s)
      self.eps_xy_a1 = self.construct_ribbon(self.gauss_amp_a1,self.gauss_along_x,
        self.gauss_along_y,self.gauss_sigma,self.center_x,self.center_y,self.width_x,
        self.width_y,self.smp,lmax,mmax,cell_s)

  def construct_wire(self,eps1,eps2,cell_s,lmax,mmax,circle=False,vertices_file=""):
    #
    # Construct the wire profile.
    # Inputs:
    # eps1, eps2: eps1 is the epsilon inside the material. eps2 outside.
    # cell_s: An object of the class cell, initialised at the beginning
    # lmax,mmax: (2*lmax+1,2*mmax+1) dimension of the wire profile
    # circle: Flag for a circular cross-section of the wire profile
    # vertices_file: File to read in the vertices of the polygon forming the
    #                cross-section of the wire
    # 
    if circle==False and vertices_file=="":
      print("Supply information on wire cross-section; circular or provide vertices file.")
      exit()
    if circle==False:
      fp = open(vertices_file,"r")
      lines = fp.readlines()
      fp.close()
  
      V = []
      for i in range(len(lines)):
        if "vertices_pos" in lines[i]:
          w = lines[i+1].split()
          w = list(filter(bool,w))
          n_v = eval(w[0])
          for j in range(n_v):
            w = lines[i+j+2].split()
            w = list(filter(bool,w))
            V.append([eval(w[0]),eval(w[1])])
          break
      V = np.array(V)
      xmin = min(V[:,0])
      ymin = min(V[:,1])
      xmax = max(V[:,0])
      ymax = max(V[:,1])
  
      bbPath = mplPath.Path(V)
      X = np.arange(0.,cell_s.celldm1,cell_s.celldm1/(2*lmax+1))
      Y = np.arange(0.,cell_s.celldm2,cell_s.celldm2/(2*mmax+1))
      eps_xy = np.zeros((2*lmax+1,2*mmax+1))
      for i in range(2*lmax+1):
        for j in range(2*mmax+1):
          if bbPath.contains_point((X[i],Y[j])):
            eps_xy[i][j] = eps1
          else:
            eps_xy[i][j] = eps2
      if ymin < 0:
        y = 0 - cell_s.celldm1/(2*mmax+1)
        cnt = 1
        while(y>=ymin):
          for i in range(len(X)):
            if not bbPath.contains_point((X[i],y)):
              eps_xy[i][2*mmax+1-cnt] = eps1
            else:
              eps_xy[i][2*mmax+1-cnt] = eps2
          cnt = cnt + 1
          y = y - cell_s.celldm1/(2*mmax+1)
    else:
      s0 = self.radius
      c_x = self.center_a1*cell_s.A2[0][0] + self.center_a2*cell_s.A2[1][0]
      c_y = self.center_a1*cell_s.A2[0][1] + self.center_a2*cell_s.A2[1][1]
      eps_xy = np.zeros((2*lmax+1,2*mmax+1))
      for l in range(2*lmax+1):
        x = l*cell_s.celldm1/(2*lmax+1)
        for m in range(2*mmax+1):
          y = m*cell_s.celldm1/(2*mmax+1)
          s = np.sqrt((x-c_x)**2 + (y-c_y)**2)
          eps_xy[l][m] = (0.5*(eps2-eps1)*erf( (s-s0)/self.smp )) - (0.5*(eps2-eps1)*erf( (s+s0)/self.smp) ) + eps2
    return eps_xy

  def construct_epsZ(self,cell_s,nmax):
    if self.Profile == "Slab":
      self.epsZ_a1 = self.construct_slab(self.eps1_a1,self.eps2_a1,cell_s,nmax)
      self.epsZ_a2 = self.construct_slab(self.eps1_a2,self.eps2_a2,cell_s,nmax)
      self.epsZ_a3 = self.construct_slab(self.eps1_a3,self.eps2_a3,cell_s,nmax)
    elif self.Profile == "Gaussian":
      self.epsZ_a1 = self.construct_gauss(self.gauss_amp_a1,self.gauss_sigma,cell_s,nmax)
      self.epsZ_a2 = self.construct_gauss(self.gauss_amp_a2,self.gauss_sigma,cell_s,nmax)
      self.epsZ_a3 = self.construct_gauss(self.gauss_amp_a3,self.gauss_sigma,cell_s,nmax)

  def construct_slab(self,eps1,eps2,cell_s,nmax):
    epsZ_tmp = []
    a3_list = np.arange(0,1,1./(2*nmax+1))
    c_crys = self.center/cell_s.celldm3
    c_crys_n = np.searchsorted(a3_list,c_crys)/float(2*nmax+1)

    # Crys -> Cartesian
    c_z = c_crys_n*cell_s.celldm3
    a3_list = a3_list*cell_s.celldm3

    # Facilitate periodic boundary conditions.
    # Checking left and right of the box along a3
    of_a3_l = False
    if c_z - self.width/2 - 2*self.smp < 0.:
      of_a3_l = True
    of_a3_r = False
    if c_z + self.width/2 + 2*self.smp > cell_s.celldm3:
      of_a3_r = True

    for a3_e in a3_list:
      if of_a3_l and cell_s.celldm3 - a3_e <  self.width/2 + 2*self.smp:
        z = 0 - (cell_s.celldm3 - a3_e)
      elif of_a3_r and a3_e <  self.width/2 + 2*self.smp:
        z = cell_s.celldm3 + a3_e
      else:
        z = a3_e
      epsZ_tmp.append( (0.5*(eps2-eps1)*erf( (z-(c_z+self.width/2.))/self.smp)) - (0.5*(eps2-eps1)*erf( (z-(c_z-self.width/2.))/self.smp) ) + eps2 )
    return epsZ_tmp

  def construct_gauss(self,amp,sig,cell_s,nmax):
    epsZ_tmp = []
    # Facilitate periodic boundary conditions.
    # Checking left and right of the box along a3
    of_a3_l = False
    if self.center - 4*sig < 0:
      of_a3_l = True
    of_a3_r = False
    if self.center + 4*sig > cell_s.celldm3:
      of_a3_r = True
    for n in range(0, 2*nmax+1):
      z = n*cell_s.celldm3/(2*nmax)
      if of_a3_l and cell_s.celldm3 - z < 4*sig:
        z = 0 - (cell_s.celldm3 - z)
      elif of_a3_r and z < 4*sig:
        z = cell_s.celldm3 + z
      epsZ_tmp.append(1.0 + amp*np.exp(-1.*(z-self.center)**2/(2*sig**2)))
    return epsZ_tmp

  def FFT(self,lmax,mmax,nmax):
    if self.Profile == "Slab" or self.Profile == "Gaussian":
      self.epsGz_a1 = self.FFT_1D(self.epsZ_a1,nmax)
      self.epsGz_a2 = self.FFT_1D(self.epsZ_a2,nmax)
      self.epsGz_a3 = self.FFT_1D(self.epsZ_a3,nmax)
    elif self.Profile == "Wire" or self.Profile == "Ribbon":
      self.eps_GxGy_a1 = self.FFT_2D(self.eps_xy_a1,lmax,mmax)
      self.eps_GxGy_a2 = self.FFT_2D(self.eps_xy_a2,lmax,mmax)
      self.eps_GxGy_a3 = self.FFT_2D(self.eps_xy_a3,lmax,mmax)

  def FFT_2D(self,F_xy, lmax, mmax):  #2D Fourier transform
    F_GxGy = np.fft.fft2(F_xy)
    F_GxGy_shift = np.fft.fftshift(F_GxGy)
    F_GxGy_req = F_GxGy_shift/np.double(2*lmax+1)/np.double(2*mmax+1)
    return F_GxGy_req

  def IFFT_2D(self,F_GxGy, lmax, mmax):   #2D Inverse fourier transform
    F_xy = np.fft.ifft2(np.fft.ifftshift(F_GxGy*np.double(2*lmax+1)*np.double(2*mmax+1)))
    return F_xy

  # 1D fourier transform
  def FFT_1D(self,F_z,nmax):
    F_Gz = np.fft.fft(F_z)
    F_Gz_shift = np.fft.fftshift(F_Gz)
    F_Gz_req = F_Gz_shift/float(2*nmax+1)
    return F_Gz_req

  # Parameters to be read from the &DIELECTRIC_PARAMETERS section
  def read_params(self,file_name):                
    fp = open(file_name,'r')
    lines = fp.readlines()
    for il in range(len(lines)):
      if "&DIELECTRIC_PARAMETERS" in lines[il]:
         w = lines[il].split()
         w = list(filter(bool,w))
         if len(w) == 1:
           print("ERROR, please specify type of profile in input: %s, line: %d"%(file_name,il+1))
         else:
           self.Profile = w[1]
         for jl in range(il+1,len(lines)):
           if "/" in lines[jl]:
             break
           if "Epsilon1_a1" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.eps1_a1 = eval(w[1])
           elif "Epsilon2_a1" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.eps2_a1 = eval(w[1])
           elif "Epsilon1_a2" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.eps1_a2 = eval(w[1])
           elif "Epsilon2_a2" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.eps2_a2 = eval(w[1])
           elif "Epsilon1_a3" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.eps1_a3 = eval(w[1])
           elif "Epsilon2_a3" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.eps2_a3 = eval(w[1])
           elif "Width" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.width = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.width = eval(w1[0])/0.529177249
             else:
               self.width = eval(w1[0])
           elif "W_x" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.width_x = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.width_x = eval(w1[0])/0.529177249
             else:
               self.width_x = eval(w1[0])
           elif "W_y" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.width_y = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.width_y = eval(w1[0])/0.529177249
             else:
               self.width_y = eval(w1[0])
           elif "Centre" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.center = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.center = eval(w1[0])/0.529177249
             else:
               self.center = eval(w1[0])
           elif "C_x" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.center_x = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.center_x = eval(w1[0])/0.529177249
             else:
               self.center_x = eval(w1[0])
           elif "C_y" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.center_y = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.center_y = eval(w1[0])/0.529177249
             else:
               self.center_y = eval(w1[0])
           elif "Gauss_amp_a1" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.gauss_amp = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.gauss_amp_a1 = eval(w1[0])/0.529177249
             else:
               self.gauss_amp_a1 = eval(w1[0])
           elif "Gauss_amp_a2" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.gauss_amp_a2 = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.gauss_amp_a2 = eval(w1[0])/0.529177249
             else:
               self.gauss_amp_a2 = eval(w1[0])
           elif "Gauss_amp_a3" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.gauss_amp_a3 = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.gauss_amp_a3 = eval(w1[0])/0.529177249
             else:
               self.gauss_amp_a3 = eval(w1[0])
           elif "Sigma" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.gauss_sigma = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.gauss_sigma = eval(w1[0])/0.529177249
             else:
               self.gauss_sigma = eval(w1[0])
           elif "Smoothness" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.smp = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.smp = eval(w1[0])/0.529177249
             else:
               self.smp = eval(w1[0])
           elif "Radius" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.radius = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.radius = eval(w1[0])/0.529177249
             else:
               self.radius = eval(w1[0])
           elif "c_a1" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.center_a1 = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.center_a1 = eval(w1[0])/0.529177249
             else:
               self.center_a1 = eval(w1[0])
           elif "c_a2" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.center_a2 = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.center_a2 = eval(w1[0])/0.529177249
             else:
               self.center_a2 = eval(w1[0])
           elif "Vertices_file" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1].split()
             w1 = list(filter(bool,w1))
             self.vertices_file = w1[0]
           elif "Circle" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.circle = eval(w[1])
           elif "Plot_eps" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.plot_eps = eval(w[1])

class gaussian:
  def __init__(self):
    self.sigma = 0.5
    self.tot_charge = 0
    self.c_a1 = 0.5
    self.c_a2 = 0.5
    self.c_a3 = 0.5
    self.rho_r = []
    self.rho_G = []
  def FFT(self,lmax,mmax,nmax):
    F_G = np.fft.fftn(self.rho_r)
    F_G_shift = np.fft.fftshift(F_G)
    self.rho_G = F_G_shift/(2.*lmax+1)/(2.*mmax+1)/(2.*nmax+1)

  def disp_params(self):
    print("GAUSSIAN_PARAMETERS:")
    print("Total charge:", self.tot_charge)
    print("Center of the gaussian (in crystal units):")
    print(self.c_a1, self.c_a2, self.c_a3)
    print("Gaussian width (bohr):")
    print(self.sigma)
    print("\n")

  def construct_rho(self,cell_s,lmax,mmax,nmax):
    a1_list = np.zeros(2*lmax+1)
    a2_list = np.zeros(2*mmax+1)
    a3_list = np.zeros(2*nmax+1)
    for l in range(2*lmax+1):
      a1_list[l] = l*1./(2*lmax+1)
    for m in range(2*mmax+1):
      a2_list[m] = m*1./(2*mmax+1)
    for n in range(2*nmax+1):
      a3_list[n] = n*1./(2*nmax+1)

    # Original center of the Gaussian, in crystal units
    c_crys = np.array([self.c_a1,self.c_a2,self.c_a3])
    
    c_a1_g = np.searchsorted(a1_list,self.c_a1)/float(2*lmax+1)
    c_a2_g = np.searchsorted(a2_list,self.c_a2)/float(2*mmax+1)
    c_a3_g = np.searchsorted(a3_list,self.c_a3)/float(2*nmax+1)
   
    # New, slightly shifted center. 
    c_crys_n = np.array([c_a1_g,c_a2_g,c_a3_g])
    # In Cartesian 
    c_x = np.dot(c_crys_n,cell_s.A2[:,0])
    c_y = np.dot(c_crys_n,cell_s.A2[:,1])
    c_z = np.dot(c_crys_n,cell_s.A2[:,2])
    
    
    
    a1_list = a1_list*cell_s.celldm1
    a2_list = a2_list*cell_s.celldm2
    a3_list = a3_list*cell_s.celldm3
    # Facilitate periodic boundary conditions.

    # Checking left and right of the box along a3
    of_a3_l = False
    c_a3 = c_crys_n[2]*cell_s.celldm3
    if c_a3 - 4*self.sigma < 0.:
      of_a3_l = True
    of_a3_r = False
    if c_a3 + 4*self.sigma > cell_s.celldm3:
      of_a3_r = True
    if of_a3_l and of_a3_r:
      print("Error: Model charge Sigma too large, spilling over!")
      sys.exit()
    # Checking left and right of the box along a1
    of_a1_l = False
    c_a1 = c_crys_n[0]*cell_s.celldm1
    if c_a1 - 4*self.sigma < 0.:
      of_a1_l = True
    of_a1_r = False
    if c_a1 + 4*self.sigma > cell_s.celldm1:
      of_a1_r = True
    if of_a1_l and of_a1_r:
      print("Error: Model charge Sigma too large, spilling over!")
      sys.exit()
    # Checking left and right of the box along a2
    of_a2_l = False
    c_a2 = c_crys_n[1]*cell_s.celldm2
    if c_a2 - 4*self.sigma < 0.:
      of_a2_l = True
    of_a2_r = False
    if c_a2 + 4*self.sigma > cell_s.celldm2:
      of_a2_r = True
    if of_a2_l and of_a2_r:
      print("Error: Model charge Sigma too large, spilling over!")
      sys.exit()
 
    # Construct rho   
    self.rho_r = []
    for a1_e in a1_list:
      temp1 = []
      for a2_e in a2_list:
        temp2 = []
        for a3_e in a3_list:
          if of_a1_l and cell_s.celldm1 - a1_e < 4*self.sigma:
            a1_el = 0 - (cell_s.celldm1 - a1_e)
          elif of_a1_r and a1_e < 4*self.sigma:
            a1_el = cell_s.celldm1 + a1_e
          else:
            a1_el = a1_e
          if of_a2_l and cell_s.celldm2 - a2_e < 4*self.sigma:
            a2_el = 0 - (cell_s.celldm2 - a2_e)
          elif of_a2_r and a2_e < 4*self.sigma:
            a2_el = cell_s.celldm2 + a2_e
          else:
            a2_el = a2_e
          x = cell_s.A[0][0]*a1_el+ cell_s.A[1][0]*a2_el
          y = cell_s.A[0][1]*a1_el + cell_s.A[1][1]*a2_el
          if of_a3_l and cell_s.celldm3 - a3_e < 4*self.sigma:
            z = 0 - (cell_s.celldm3 - a3_e)
          elif of_a3_r and a3_e < 4*self.sigma:
            z = cell_s.celldm3 + a3_e
          else:
            z = a3_e
          temp2.append( np.exp(-( (x-c_x)**2 + (y-c_y)**2 + (z-c_z)**2 )/2./self.sigma**2 )/self.sigma**3/(2*np.pi)**1.5 )
    
        temp1.append(temp2)
      self.rho_r.append(temp1)    
    self.rho_r = np.array(self.rho_r)*(self.tot_charge)


# Parameters to be read from the &GAUSSSIAN_PARAMETERS section    

  def read_params(self,file_name):
    fp = open(file_name,'r')
    lines = fp.readlines()
    for il in range(len(lines)):
      if "&GAUSSIAN_PARAMETERS" in lines[il]:
         for jl in range(il+1,len(lines)):
           if "/" in lines[jl]:
             break
           if "Total_charge" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.tot_charge = eval(w[1])
           if "Sigma" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             w1 = w[1]
             w1 = w1.split()
             w1 = list(filter(bool,w1))
             if len(w1) > 1:
               if w1[1] == "bohr":
                 self.sigma = eval(w1[0])
               elif w1[1] == "angstrom":
                 self.sigma = eval(w1[0])/0.529177249
             else:
               self.sigma = eval(w1[0])
           if "Centre_a1" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.c_a1 = eval(w[1])
           if "Centre_a2" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.c_a2 = eval(w[1])
           if "Centre_a3" in lines[jl]:
             w = lines[jl].split("=")
             if len(w) != 2:
               print("ERROR while parsing input file: %s, line: %d"%(file_name,jl+1))
               sys.exit()
             self.c_a3 = eval(w[1])

class GlobalValues:
  i_curr = 0
  j_curr = 0
  k_curr = int(0)
  c_g = cell()
  eps_g = diel_profile()
  kmax = 0 
  lmax = 0 
  mmax = 0 


