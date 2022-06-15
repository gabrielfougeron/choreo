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

__version__ = "0.1.0"

# To buid and use inplace, run the following command :
# python setup.py build_ext --inplace

# To build for the current platform, run :
# python setup.py bdist_wheel


# os.environ["CC"] = "gcc"
# os.environ["CC"] = "icc"

extra_compile_args = ["-O2"]

extra_link_args = []


extension = Extension(
    name = "choreo.Choreo_cython_funs",
    sources =  ["choreo/Choreo_cython_funs.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args,
                  )

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
