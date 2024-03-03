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

cython_extnames_safemath = [
    ("choreo.cython.funs"                   , False),
    ("choreo.cython._ActionSym"             , False),
    ("choreo.cython._NBodySyst"             , False),
    ("choreo.cython.funs_serial"            , False),
    ("choreo.cython.test_blas"              , False),
    ("choreo.scipy_plus.cython.ODE"         , False),
    ("choreo.scipy_plus.cython.SegmQuad"    , False),
    ("choreo.scipy_plus.cython.test"        , False),
    ("choreo.scipy_plus.cython.blas_consts" , False),
    ("choreo.scipy_plus.cython.eft_lib"     , True ),
]

cython_extnames = [item[0] for item in cython_extnames_safemath]
cython_safemath_needed = [item[1] for item in cython_extnames_safemath]


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
    
    ignore_warnings_args = [
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-incompatible-pointer-types-discards-qualifiers",
        "-Wno-unused-command-line-argument"
    ] 

    if ("PYODIDE" in os.environ): # Building for Pyodide

        extra_compile_args_std = ["-O3","-ffast-math","-flto",  *ignore_warnings_args]
        extra_compile_args_safe = ["-O3","-flto",  *ignore_warnings_args]
        extra_link_args = ["-flto", ]

    else:

        # all_compilers = ['icx','clang','gcc']
        all_compilers = ['clang']
        # all_compilers = ['gcc']

        for compiler in all_compilers:

            if not(shutil.which(compiler) is None):

                os.environ['CC'] = compiler
                os.environ['LDSHARED'] = compiler+' -shared'
                
                break
# 
        extra_compile_args_std = ["-O0","-march=native", "-fopenmp", "-lm", *ignore_warnings_args]
        extra_compile_args_safe = ["-O0", "-fopenmp", "-lm", *ignore_warnings_args]
        extra_link_args = ["-fopenmp", "-lm",]

        # extra_compile_args_std = ["-Ofast", "-march=native", "-fopenmp", "-lm", "-flto", *ignore_warnings_args]
        # extra_compile_args_safe = ["-O3", "-fopenmp", "-lm", "-flto", *ignore_warnings_args]
        # extra_link_args = ["-fopenmp", "-lm", "-flto",  *ignore_warnings_args]


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

##### Profiler only ####
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


