'''
Creates anc compiles C code from Cython file

'''

import os
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# to buid, run the following command :
# python setup.py build_ext --inlpace

os.environ["CC"] = "gcc"
# ~ os.environ["CC"] = "icc"


# ~ extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ]
# ~ extra_compile_args = ["-O2", "-march=native", "-fopenmp" ]
# ~ extra_compile_args = ["-O3", "-march=native", "-fopenmp" ]
extra_compile_args = ["-Ofast", "-march=native", "-fopenmp" ]
# ~ extra_compile_args = ["-O0", "-march=native", "-fopenmp" ]

extra_link_args = []
# ~ extra_link_args = ["-lm"]



# ~ os.utime('Choreo_cython_funs.pyx',None)


extension = Extension("Choreo_cython_funs", ["Choreo_cython_funs.pyx"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args = extra_compile_args,
                  extra_link_args = extra_link_args,
                  )

setup(
    ext_modules = cythonize(extension, language_level = "3",annotate=True)
)
