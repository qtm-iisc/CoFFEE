from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy, os

ext_modules=[ Extension("matvec2D",
              ["matvec2D.pyx"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "matvec2D",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs = [numpy.get_include()] 
)
os.system("mv matvec2D*.so matvec2D.so")
