'''
Creates and compiles C code from Cython file

'''

import os
import sys
import sysconfig
import shutil
import setuptools
import numpy
import platform

try:
    import pyfftw
    PYFFTW_AVAILABLE_COMPILE = True
except:
    PYFFTW_AVAILABLE_COMPILE = False

import Cython.Build
import Cython.Compiler
Cython.Compiler.Options.cimport_from_pyx = True # Mandatory for scipy.LowLevelCallable.from_cython
Cython.Compiler.Options.fast_fail = True
Cython.warn.undeclared = True
src_ext = '.pyx'

cython_extnames_safemath = [
    ("choreo.cython._ActionSym"                 , False),
    ("choreo.cython._NBodySyst"                 , False),
    ("choreo.cython._NBodySyst_ann"             , False),
    ("choreo.cython.pyfftw_fake"                , False),
    ("choreo.segm.cython.ODE"                   , False),
    ("choreo.segm.cython.quad"                  , False),
    ("choreo.segm.cython.test"                  , False),
    ("choreo.segm.cython.eft_lib"               , True ),
    ("choreo.scipy_plus.cython.blas_consts"     , False),
    ("choreo.scipy_plus.cython.blas_cheatsheet" , False),
    ("choreo.scipy_plus.cython.kepler"          , False),
    ("choreo.scipy_plus.cython.misc"            , False),
]

cython_extnames = [item[0] for item in cython_extnames_safemath]
cython_safemath_needed = [item[1] for item in cython_extnames_safemath]

for opt_key in ['profile','0','1','2','3','fast']:
    
    cmdline_opt = f"-O{opt_key}"
    
    if cmdline_opt in sys.argv:
        opt_lvl = opt_key
        sys.argv.remove(cmdline_opt)
        break

else:
    opt_lvl = 'fast'
    # opt_lvl = '0'

if platform.system() == "Windows":
    
    ignore_warnings_args = [
        # "-Wno-unused-variable",
        # "-Wno-unused-function",
        # "-Wno-incompatible-pointer-types-discards-qualifiers",
        # "-Wno-unused-command-line-argument"
    ] 

    extra_compile_args_std = {
        "profile" : ["/Od", "/openmp", *ignore_warnings_args],
        "0" : ["/Od", "/openmp", *ignore_warnings_args],
        "1" : ["/Ox", "/openmp", *ignore_warnings_args],
        "2" : ["/O2", "/openmp", *ignore_warnings_args],
        "3" : ["/O2", "/openmp", *ignore_warnings_args],
        "fast" : ["/O2", "/GL", "/openmp", *ignore_warnings_args],
    }[opt_lvl]
    
    extra_compile_args_safe = {
        "profile" : ["/Od", "/openmp", *ignore_warnings_args],
        "0" : ["/Od", "/openmp", *ignore_warnings_args],
        "1" : ["/Ox", "/openmp", *ignore_warnings_args],
        "2" : ["/O2", "/openmp", *ignore_warnings_args],
        "3" : ["/O2", "/openmp", *ignore_warnings_args],
        "fast" : ["/O2", "/GL","/openmp", *ignore_warnings_args],
    }[opt_lvl]
    
    extra_link_args = {
        "profile" : [*ignore_warnings_args],
        "0" : [*ignore_warnings_args],
        "1" : [*ignore_warnings_args],
        "2" : [*ignore_warnings_args],
        "3" : [*ignore_warnings_args],
        "fast" : ["/GL", *ignore_warnings_args],
    }[opt_lvl]

elif platform.system() == "Darwin": # MacOS

    ignore_warnings_args = [
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-incompatible-pointer-types-discards-qualifiers",
        "-Wno-unused-command-line-argument"
    ] 
    
    std_args = ["-march=native", "-Xpreprocessor", "-std=c99", "-lm"]
    std_link_args = ["-lm", "-lomp"]

    extra_compile_args_std = {
        "profile" : ["-Og", *std_args, *ignore_warnings_args],
        "0" : ["-O0", *std_args, *ignore_warnings_args],
        "1" : ["-O1", *std_args, *ignore_warnings_args],
        "2" : ["-O2", *std_args, *ignore_warnings_args],
        "3" : ["-O3", *std_args, *ignore_warnings_args],
        "fast" : ["-Ofast", "-flto", *std_args, *ignore_warnings_args],
    }[opt_lvl]
    
    extra_compile_args_safe = {
        "profile" : ["-Og", *std_args, *ignore_warnings_args],
        "0" : ["-O0", *std_args, *ignore_warnings_args],
        "1" : ["-O1", *std_args, *ignore_warnings_args],
        "2" : ["-O2", *std_args, *ignore_warnings_args],
        "3" : ["-O3", *std_args, *ignore_warnings_args],
        "fast" : ["-O3", "-flto", *std_args, *ignore_warnings_args],
    }[opt_lvl]

    extra_link_args = {
        "profile" : [*std_link_args, *ignore_warnings_args],
        "0" : [*std_link_args, *ignore_warnings_args],
        "1" : [*std_link_args, *ignore_warnings_args],
        "2" : [*std_link_args, *ignore_warnings_args],
        "3" : [*std_link_args, *ignore_warnings_args],
        "fast" : ["-flto", *std_link_args, *ignore_warnings_args],
    }[opt_lvl]

elif platform.system() == "Linux":
    
    ignore_warnings_args = [
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-incompatible-pointer-types-discards-qualifiers",
        "-Wno-unused-command-line-argument"
    ] 

    if ("PYODIDE" in os.environ): # Building for Pyodide

        extra_compile_args_std = {
            "profile" : ["-Og",  *ignore_warnings_args],
            "0" : ["-O0",  *ignore_warnings_args],
            "1" : ["-O1",  *ignore_warnings_args],
            "2" : ["-O2",  *ignore_warnings_args],
            "3" : ["-O3",  *ignore_warnings_args],
            "fast" : ["-O3","-ffast-math","-flto",  *ignore_warnings_args],
        }[opt_lvl]
        
        extra_compile_args_safe = {
            "profile" : ["-Og",  *ignore_warnings_args],
            "0" : ["-O0",  *ignore_warnings_args],
            "1" : ["-O1",  *ignore_warnings_args],
            "2" : ["-O2",  *ignore_warnings_args],
            "3" : ["-O3",  *ignore_warnings_args],
            "fast" : ["-O3","-flto",  *ignore_warnings_args],
        }[opt_lvl]

        extra_link_args = {
            "profile" : [*ignore_warnings_args],
            "0" : [*ignore_warnings_args],
            "1" : [*ignore_warnings_args],
            "2" : [*ignore_warnings_args],
            "3" : [*ignore_warnings_args],
            "fast" : ["-flto", *ignore_warnings_args],
        }[opt_lvl]

    else:
        
        std_args = ["-march=native", "-fopenmp", "-lm"]
        std_link_args = ["-lm", "-fopenmp"]

        extra_compile_args_std = {
            "profile" : ["-Og", *std_args, *ignore_warnings_args],
            "0" : ["-O0", *std_args, *ignore_warnings_args],
            "1" : ["-O1", *std_args, *ignore_warnings_args],
            "2" : ["-O2", *std_args, *ignore_warnings_args],
            "3" : ["-O3", *std_args, *ignore_warnings_args],
            "fast" : ["-Ofast", "-flto", *std_args, *ignore_warnings_args],
        }[opt_lvl]

        extra_compile_args_safe = {
            "profile" : ["-Og", "-ffp-contract=on", *std_args, *ignore_warnings_args],
            "0" : ["-O0", "-ffp-contract=on", *std_args, *ignore_warnings_args],
            "1" : ["-O1", "-ffp-contract=on", *std_args, *ignore_warnings_args],
            "2" : ["-O2", "-ffp-contract=on", *std_args, *ignore_warnings_args],
            "3" : ["-O3", "-ffp-contract=on", *std_args, *ignore_warnings_args],
            "fast" : ["-O3", "-ffp-contract=on", "-flto", *std_args, *ignore_warnings_args],
        }[opt_lvl]

        extra_link_args = {
            "profile" : [*std_link_args, *ignore_warnings_args],
            "0" : [*std_link_args, *ignore_warnings_args],
            "1" : [*std_link_args, *ignore_warnings_args],
            "2" : [*std_link_args, *ignore_warnings_args],
            "3" : [*std_link_args, *ignore_warnings_args],
            "fast" : ["-flto", *std_link_args, *ignore_warnings_args],
        }[opt_lvl]

else:

    raise ValueError(f"Unsupported platform: {platform.system()}")

cython_filenames = [ ext_name.replace('.','/') + src_ext for ext_name in cython_extnames]

# Special rule for the optional run dependency PyFFTW, disabled in pyodide
cython_extnames.append("choreo.cython.optional_pyfftw")
cython_safemath_needed.append(False)

if '--no-fftw' in sys.argv:
    sys.argv.remove('--no-fftw')
    include_pyfftw = False
else:
    include_pyfftw = PYFFTW_AVAILABLE_COMPILE and not("PYODIDE" in os.environ)

cython_filenames.append(f"choreo.cython.optional_pyfftw_{include_pyfftw}".replace('.','/') + src_ext)

optional_pyfftw_pxd_path = "choreo/cython/optional_pyfftw.pxd"
if include_pyfftw:
    pyfftw_pxd_str = """
cimport pyfftw
    """
else:
    pyfftw_pxd_str = """
cimport choreo.cython.pyfftw_fake as pyfftw
    """
if os.path.isfile(optional_pyfftw_pxd_path):
    with open(optional_pyfftw_pxd_path, "r") as text_file:
        write_optional_pyfftw_pxd = (pyfftw_pxd_str != text_file.read())
            
else:
    write_optional_pyfftw_pxd = True
    
if write_optional_pyfftw_pxd:    
    with open(optional_pyfftw_pxd_path, "w") as text_file:
        text_file.write(pyfftw_pxd_str)

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if platform.system() == "Windows":
    define_macros.append(("FFTW_NO_Complex", 1))
    define_macros.append(("CYTHON_CCOMPLEX", 0))

if opt_lvl == "fast":
    define_macros.append(("CYTHON_WITHOUT_ASSERTIONS", 1))
    define_macros.append(("CYTHON_CLINE_IN_TRACEBACK", 0))

compiler_directives = {
    'cpow' : True   ,
}

if opt_lvl == "profile" : 

    profile_compiler_directives = {
        'profile': True             ,
        'linetrace': True           ,
        'binding': True             ,
        'embedsignature' : True     ,
        'emit_code_comments' : True , 
    }
    compiler_directives.update(profile_compiler_directives)
    
    profile_define_macros = [
        ('CYTHON_TRACE', '1')       ,
        ('CYTHON_TRACE_NOGIL', '1') ,
    ]
    define_macros.extend(profile_define_macros)

else:
    
    compiler_directives.update({
        'wraparound': False         ,
        'boundscheck': False        ,
        'nonecheck': False          ,
        'initializedcheck': False   ,
        'overflowcheck': False      ,
        'overflowcheck.fold': False ,
        'infer_types': True         ,
        'binding' : False           , 
    })

include_dirs = [
    numpy.get_include(),
]

# Path must be relative for auto inclusion in the manifest
if platform.system() == "Windows":
    include_dirs.append(os.path.relpath(os.path.join(os.getcwd(), 'include', 'win')))
else:
    include_dirs.append(os.path.relpath(os.path.join(os.getcwd(), 'include')))

ext_modules = [
    setuptools.Extension(
        name = name,
        sources =  [source],
        include_dirs = include_dirs,
        extra_compile_args = extra_compile_args_safe if safemath_needed else extra_compile_args_std,
        extra_link_args = extra_link_args,
        define_macros  = define_macros ,
    )
    for (name, source, safemath_needed) in zip(cython_extnames, cython_filenames, cython_safemath_needed, strict = True)
]

if platform.system() == "Windows":
    nthreads = 0
else:
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

package_data = {key : ['*.h','*.pyx','*.pxd'] for key in packages}
exclude_package_data = {key : ['*.c'] for key in packages}

# If Pyodide wheel was generated, then copy it!
basedir = os.path.dirname(__file__)
distdir = os.path.join(basedir, "dist")

if os.path.isdir(distdir):
    
    for filename in os.listdir(os.path.join(basedir, "dist")):
        basename, ext = os.path.splitext(filename)
        
        if 'pyodide' in basename and ext == '.whl':
            
            src = os.path.join(basedir, "dist", filename)
            dstdir = os.path.join(basedir, "choreo-GUI", "choreo_GUI", "python_dist")
            dst = os.path.join(dstdir, filename)
            
            if os.path.isdir(dstdir):
                if os.path.isfile(src):
                        
                    if os.path.isfile(dst):
                        os.remove(dst)
                    
                    shutil.copyfile(src, dst)

setuptools.setup(
    ext_modules = ext_modules                   ,
    zip_safe = False                            ,
    packages = packages                         ,
    package_data = package_data                 ,
    exclude_package_data = exclude_package_data ,
)
