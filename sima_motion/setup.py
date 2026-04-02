"""Build configuration for Cython extensions."""

from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    ext_modules = cythonize([
        Extension(
            "sima_motion._motion",
            ["sima_motion/_motion.pyx"],
            include_dirs=[np.get_include()],
        ),
    ])
except ImportError:
    ext_modules = [
        Extension(
            "sima_motion._motion",
            ["sima_motion/_motion.c"],
            include_dirs=[np.get_include()],
        ),
    ]

setup(ext_modules=ext_modules)
