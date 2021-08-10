from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy, os

ext_modules=[ Extension("ps3d",
              ["ps3d.pyx"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "ps3d",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs = [numpy.get_include()] 
)

os.system("mv ps3d*.so ps3d.so")
