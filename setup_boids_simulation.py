# setup_boids_sumulation.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("boids_simulation.pyx"),
    include_dirs=[numpy.get_include()]
)