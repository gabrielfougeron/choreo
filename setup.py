'''
Creates and compiles C code from Cython file

'''

import os
import sys
import shutil
import setuptools
import numpy
import platform

# use_Cython = False
use_Cython = True

if use_Cython:
    import Cython.Build
    import Cython.Compiler
    Cython.Compiler.Options.cimport_from_pyx = True
    Cython.Compiler.Options.fast_fail = True
    src_ext = '.pyx'
else:
    src_ext = '.c'

cython_extnames = [
    "choreo.cython.funs",
    "choreo.cython.funs_new",
    "choreo.cython.funs_serial",
    "choreo.scipy_plus.cython.ODE",
    "choreo.scipy_plus.cython.SegmQuad",
    "choreo.scipy_plus.cython.test",
]

cython_safemath_needed = [
    False,
    False,
    False,
    False,
    False,
    False,
]

if platform.system() == "Windows":

    extra_compile_args_std = ["/O2", "/openmp"]
    extra_compile_args_safe = ["/O2", "/openmp"]
    extra_link_args = ["/openmp"]

    cython_extnames.append("choreo.cython.funs_parallel")
    cython_safemath_needed.append(False)

elif platform.system() == "Darwin": # MacOS

    extra_compile_args_std = ["-Ofast","-march=native", "-fopenmp"]
    extra_compile_args_safe = ["-O2", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

    cython_extnames.append("choreo.cython.funs_parallel")
    cython_safemath_needed.append(False)


elif platform.system() == "Linux":

    # print(platform.system())
    # print( os.environ)

    if ("PYODIDE" in os.environ): # Building for Pyodide

        extra_compile_args_std = ["-O3"]
        extra_compile_args_safe = ["-O2"]
        extra_link_args = []

    else:

        all_compilers = ['icx','clang','gcc']
        # all_compilers = ['clang']
        # all_compilers = ['gcc']

        for compiler in all_compilers:

            if not(shutil.which(compiler) is None):

                os.environ['CC'] = compiler
                os.environ['LDSHARED'] = compiler+' -shared'

                # print(f'Compiler: {compiler}')

                break

        # extra_compile_args_std = ["-O0","-march=native", "-fopenmp"]
        # extra_compile_args_safe = ["-O0", "-fopenmp"]
        # extra_link_args = ["-fopenmp"]

        extra_compile_args_std = ["-Ofast","-march=native", "-fopenmp"]
        extra_compile_args_safe = ["-O2", "-fopenmp"]
        extra_link_args = ["-fopenmp"]

        # extra_compile_args_std = ["-fast","-march=native", "-fopenmp"]
        # extra_compile_args_safe = ["-O2", "-fopenmp"]
        # extra_link_args = ["-fopenmp","-ipo"]

        cython_extnames.append("choreo.cython.funs_parallel")
        cython_safemath_needed.append(False)

else:

    raise ValueError(f"Unsupported platform: {platform.system()}")

cython_filenames = [ ext_name.replace('.','/') + src_ext for ext_name in cython_extnames]

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

compiler_directives = {
    'wraparound': False,
    'boundscheck': False,
    'nonecheck': False,
    'initializedcheck': False,
    'overflowcheck': False,
    'overflowcheck.fold': False,
    'infer_types': True,
}

# Profiler only ####
# profile_compiler_directives = {
#     'profile': True,
#     'linetrace': True,
#     'binding': True,
# }
# compiler_directives.update(profile_compiler_directives)
# profile_define_macros = [
#     ('CYTHON_TRACE', '1')   ,
#     ('CYTHON_TRACE_NOGIL', '1')   ,
# ]
# define_macros.extend(profile_define_macros)


include_dirs = [numpy.get_include()]

ext_modules = [
    setuptools.Extension(
        name = name,
        sources =  [source],
        include_dirs = include_dirs,
        extra_compile_args = extra_compile_args_safe if safemath_needed else extra_compile_args_std,
        extra_link_args = extra_link_args,
        define_macros  = define_macros ,
    )
    for (name,source,safemath_needed) in zip(cython_extnames,cython_filenames,cython_safemath_needed, strict = True)
]

if use_Cython:
    
    import multiprocessing
    nthreads = multiprocessing.cpu_count()

    ext_modules = Cython.Build.cythonize(
        ext_modules,
        language_level = "3",
        annotate = True,
        compiler_directives = compiler_directives,
        nthreads = nthreads,
        force = ("-f" in sys.argv),
    )
    

packages = setuptools.find_packages()

package_data = {key : ['*.h','*.c','*.pyx','*.pxd'] for key in packages}

setuptools.setup(
    platforms = ['any'],
    ext_modules = ext_modules,
    zip_safe = False,
    packages = packages,
    package_data = package_data,
    provides = ['choreo'],
)


