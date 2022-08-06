'''
Creates anc compiles C code from Cython file

'''

import os
from setuptools import find_packages,setup
# from distutils.core import setup, Extension
from distutils.core import Extension
from distutils.command.build import build as build_orig
from Cython.Build import cythonize
import numpy
import platform

__version__ = "0.1.0"

# To buid and use inplace, run the following command :
# python setup.py build_ext --inplace

# To build for the current platform, run :
# python setup.py bdist_wheel

if platform.system() == "Windows":

    extra_compile_args = ["/O2"]

else:
    # extra_compile_args = ["-O2"]
    # extra_compile_args = ["-O2","-march=native"]
    # extra_compile_args = ["-O3","-ffast-math","-march=native"]
    extra_compile_args = ["-O3","-march=native"]

extra_link_args = []

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


extension = [
    Extension(
    name = "choreo.Choreo_cython_funs",
    sources =  ["choreo/Choreo_cython_funs.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args,
    define_macros  = define_macros ,
    ),
    Extension(
    name = "choreo.Choreo_cython_scipy_plus",
    sources =  ["choreo/Choreo_cython_scipy_plus.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args,
    define_macros  = define_macros ,
    ),
]

setup(
    name = "choreo",
    author = "Gabriel Fougeron <gabriel.fougeron@hotmail.fr>",
    url = "https://github.com/gabrielfougeron/Choreographies2",
    version = __version__,
    description='',
    license = "BSD 2-Clause License",
    platforms=['any'],
    packages = find_packages(),
    ext_modules = cythonize(extension, language_level = "3",annotate=True),
    zip_safe=False,
    package_data={"choreo": ["choreo.h"]},
    provides=['choreo'],
)
