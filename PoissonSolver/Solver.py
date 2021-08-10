from .PS_main import PS_2D, PS_3D,PS_1D,PS_3D_new

def Solver(cell_s,charge_s,eps_s,imax,jmax,kmax,comm):
  if eps_s.Profile == "Slab" or eps_s.Profile == "Gaussian":
     # Call the Poisson Solver for slab
     V_r = PS_2D(cell_s,charge_s,eps_s,imax,jmax,kmax,comm)
     return V_r
  elif eps_s.Profile == "Wire" or eps_s.Profile == "Ribbon":
     # Call the Poisson Solver for wire
     V_r = PS_1D(cell_s,charge_s,eps_s,imax,jmax,kmax,comm)
     return V_r
  elif eps_s.Profile == "Bulk":
     # Call the Poisson Solver for bulk system
     V_r = PS_3D_new(cell_s,charge_s,eps_s,imax,jmax,kmax,comm)
     return V_r
