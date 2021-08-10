from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

ext_modules=[ Extension("matvec1D",
              ["matvec1D.pyx"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "matvec1D",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs = [numpy.get_include()] 
)

os.system("mv matvec1D*.so matvec1D.so")
