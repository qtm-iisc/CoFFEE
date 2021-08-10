import numpy as np
def construct_eps(cell_s,eps_s,imax,jmax,kmax):
  if eps_s.Profile == "Slab" or eps_s.Profile == "Gaussian":
    # Construct epsilon profile and fourier transform
    eps_s.construct_epsZ(cell_s,2*kmax)
    eps_s.FFT(2*imax,2*jmax,2*kmax)

    # Write profiles to file:
    if eps_s.plot_eps:
      np.save("epsZ_a1",eps_s.epsZ_a1)
      np.save("epsZ_a2",eps_s.epsZ_a2)
      np.save("epsZ_a3",eps_s.epsZ_a3)
  elif eps_s.Profile == "Wire" or eps_s.Profile == "Ribbon":
    # Construct epsilon profile and fourier transform
    eps_s.construct_epsxy(cell_s,2*imax,2*jmax)
    eps_s.FFT(2*imax,2*jmax,2*kmax)
    # Write the profiles to file
    if eps_s.plot_eps:
      np.save("eps_xy_a1",eps_s.eps_xy_a1)
      np.save("eps_xy_a2",eps_s.eps_xy_a2)
      np.save("eps_xy_a3",eps_s.eps_xy_a3)
