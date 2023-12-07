from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# Set up the package
setup(
    name="BoidsSimOMP",
    include_dirs=[np.get_include()],
    ext_modules=cythonize(
        [
            Extension(
                "BoidsSimOMP",
                ["BoidsSimOMP.pyx"],
                extra_compile_args=['/openmp'],
                extra_link_args=['/openmp'],
            )
        ],
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'overflowcheck': False,
            'nonecheck': False,
        },
    ),
)
