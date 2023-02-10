'''
Creates and compiles C code from Cython file

'''

import os
import setuptools
import distutils # Deprecation of distutils. See  https://peps.python.org/pep-0632/
import Cython.Build
import numpy
import platform

__version__ = "0.2.0"

# To buid and use inplace, run the following command :
# python setup.py build_ext --inplace

# To build for the current platform, run :
# python setup.py bdist_wheel

if platform.system() == "Windows":

    extra_compile_args = ["/O2"]

else:

    # print(platform.system())
    # print( os.environ)

    if ("PYODIDE_ROOT" in os.environ): # Building for Pyodide

        extra_compile_args = ["-O3"]

    elif not(distutils.spawn.find_executable('clang') is None):

        os.environ['CC'] = 'clang'
        os.environ['LDSHARED'] = 'clang -shared'

        # extra_compile_args = ["-O2","-march=native"]
        # extra_compile_args = ["-O3","-march=native"]
        extra_compile_args = ["-Ofast","-march=native"]
        # extra_compile_args = []

#     elif not(distutils.spawn.find_executable('icc') is None):
# 
#         os.environ['CC'] = 'icc'
#         os.environ['LDSHARED'] = 'icc -shared'
# 
#         extra_compile_args = ["-Ofast","-march=native","-w","-no-multibyte-chars"]

    else:

        extra_compile_args = ["-O3","-march=native"]
    


extra_link_args = []

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


cython_extnames = [
    "choreo.Choreo_cython_funs_2D",
    "choreo.Choreo_cython_funs",
    "choreo.Choreo_cython_scipy_plus",
]

cython_filenames = [
    "choreo/Choreo_cython_funs_2D.pyx",
    "choreo/Choreo_cython_funs.pyx",
    "choreo/Choreo_cython_scipy_plus.pyx",
]

compiler_directives = {
    'wraparound': False,
    # 'boundscheck': False,
    # 'nonecheck': False,
    # 'initializedcheck': False,
    # 'overflowcheck': False,
    # 'overflowcheck.fold': False,
    # 'infer_types': True,
}

# #### Profiler only ####
# profile_compiler_directives = {
#     'profile': True,
#     'linetrace': True,
#     'binding': True,
# }
# compiler_directives.update(profile_compiler_directives)
# 
# profile_define_macros = [('CYTHON_TRACE', '1')]
# define_macros.extend(profile_define_macros)




extensions = [
    distutils.core.Extension(
    name = name,
    sources =  [source],
    include_dirs=[numpy.get_include()],
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args,
    define_macros  = define_macros ,
    )
    for (name,source) in zip(cython_extnames,cython_filenames)
]

ext_modules = Cython.Build.cythonize(
    extensions,
    language_level = "3",
    annotate = True,
    force = True,
    compiler_directives = compiler_directives,
)

setuptools.setup(
    name = "choreo",
    author = "Gabriel Fougeron <gabriel.fougeron@hotmail.fr>",
    url = "https://github.com/gabrielfougeron/choreo",
    version = __version__,
    description="A set of tools to compute periodic solutions to the Newtonian N-body problem",
    license = "BSD 2-Clause License",
    platforms=['any'],
    packages = setuptools.find_packages(),
    ext_modules = ext_modules,
    zip_safe=False,
    package_data={"choreo": ["choreo.h"]},
    provides=['choreo'],
)
