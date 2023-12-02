from distutils.core import setup
import numpy as np
from Cython.Build import cythonize

# Set up the package
setup(name="BoidsSimCython",
      include_dirs=[np.get_include()],
    ext_modules=cythonize("BoidsSimCython.pyx"),
)
