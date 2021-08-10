import os
path = ["PoissonSolver/matvec1D.so","PoissonSolver/matvec2D.so","PoissonSolver/ps3d.so"]
for p in path:
  if os.path.isfile(p):
    os.remove(p)
print(".so files removed")
