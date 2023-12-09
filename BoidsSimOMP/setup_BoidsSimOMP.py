from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "BoidsSimOMP",
        ["BoidsSimOMP.pyx"],
        
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

# Set up the package
setup(
    name="BoidsSimOMP",
    include_dirs=[np.get_include()],
    ext_modules=cythonize(
        [
            Extension(
                "BoidsSimOMP",
                ["BoidsSimOMP.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
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
print("CYTHONISED!")
