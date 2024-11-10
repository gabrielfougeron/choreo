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
    PYFFTW_AVAILABLE = True
except:
    PYFFTW_AVAILABLE = False

import Cython.Build
import Cython.Compiler
Cython.Compiler.Options.cimport_from_pyx = True # Mandatory for scipy.LowLevelCallable.from_cython
Cython.Compiler.Options.fast_fail = True
Cython.warn.undeclared = True
src_ext = '.pyx'

cython_extnames_safemath = [
    ("choreo.cython._ActionSym"             , False),
    ("choreo.cython._NBodySyst"             , False),
    ("choreo.cython.pyfftw_fake"            , False),
    ("choreo.scipy_plus.cython.ODE"         , False),
    ("choreo.scipy_plus.cython.SegmQuad"    , False),
    ("choreo.scipy_plus.cython.test"        , False),
    ("choreo.scipy_plus.cython.blas_consts" , False),
    ("choreo.scipy_plus.cython.eft_lib"     , True ),
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
    
    # 
    # print(f'{os.environ['CC'] = }')
    # 
    #     os.environ['CC'] = compiler
    #     os.environ['LDSHARED'] = compiler+' -shared'
    #     
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

    # cython_extnames.append("choreo.cython.funs_parallel")
    # cython_safemath_needed.append(False)

elif platform.system() == "Darwin": # MacOS
    
    os.environ['CC'] = "clang"
    os.environ['LDSHARED'] = 'clang -shared'
    
    ignore_warnings_args = [
        "-Wno-unused-variable",
        "-Wno-unused-function",
        "-Wno-incompatible-pointer-types-discards-qualifiers",
        "-Wno-unused-command-line-argument"
    ] 

    extra_compile_args_std = {
        "profile" : ["-Og", "-fopenmp", "-lm", *ignore_warnings_args],
        "0" : ["-O0", "-fopenmp", "-lm", *ignore_warnings_args],
        "1" : ["-O1", "-fopenmp", "-lm", *ignore_warnings_args],
        "2" : ["-O2", "-fopenmp", "-lm", *ignore_warnings_args],
        "3" : ["-O3", "-fopenmp", "-lm", *ignore_warnings_args],
        "fast" : ["-Ofast", "-fopenmp", "-lm", "-flto", *ignore_warnings_args],
    }[opt_lvl]
    
    extra_compile_args_safe = {
        "profile" : ["-Og", "-fopenmp", "-lm", *ignore_warnings_args],
        "0" : ["-O0", "-fopenmp", "-lm", *ignore_warnings_args],
        "1" : ["-O1", "-fopenmp", "-lm", *ignore_warnings_args],
        "2" : ["-O2", "-fopenmp", "-lm", *ignore_warnings_args],
        "3" : ["-O3", "-fopenmp", "-lm", *ignore_warnings_args],
        "fast" : ["-O3", "-fopenmp", "-lm", "-flto", *ignore_warnings_args],
    }[opt_lvl]

    extra_link_args = {
        "profile" : ["-fopenmp", "-lm", *ignore_warnings_args],
        "0" : ["-fopenmp", "-lm", *ignore_warnings_args],
        "1" : ["-fopenmp", "-lm", *ignore_warnings_args],
        "2" : ["-fopenmp", "-lm", *ignore_warnings_args],
        "3" : ["-fopenmp", "-lm", *ignore_warnings_args],
        "fast" : ["-fopenmp", "-lm", "-flto", *ignore_warnings_args],
    }[opt_lvl]

#     cython_extnames.append("choreo.cython.funs_parallel")
#     cython_safemath_needed.append(False)

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

        # all_compilers = ['icx','clang','gcc']
        # all_compilers = ['clang']
        all_compilers = ['gcc']
        # all_compilers = ['icx'] 

        for compiler in all_compilers:

            if not(shutil.which(compiler) is None):

                os.environ['CC'] = compiler
                os.environ['LDSHARED'] = compiler+' -shared'
                
                break

        extra_compile_args_std = {
            "profile" : ["-Og", "-fopenmp", "-lm", *ignore_warnings_args],
            "0" : ["-O0", "-fopenmp", "-lm", *ignore_warnings_args],
            "1" : ["-O1", "-fopenmp", "-lm", *ignore_warnings_args],
            "2" : ["-O2", "-march=native", "-fopenmp", "-lm", *ignore_warnings_args],
            "3" : ["-O3", "-march=native", "-fopenmp", "-lm", *ignore_warnings_args],
            "fast" : ["-Ofast", "-march=native", "-fopenmp", "-lm", "-flto", *ignore_warnings_args],
        }[opt_lvl]
        
        extra_compile_args_safe = {
            "profile" : ["-Og", "-fopenmp", "-lm", *ignore_warnings_args],
            "0" : ["-O0", "-fopenmp", "-lm", *ignore_warnings_args],
            "1" : ["-O1", "-fopenmp", "-lm", *ignore_warnings_args],
            "2" : ["-O2", "-march=native", "-fopenmp", "-lm", *ignore_warnings_args],
            "3" : ["-O3", "-march=native", "-fopenmp", "-lm", *ignore_warnings_args],
            "fast" : ["-O3", "-march=native", "-fopenmp", "-lm", "-flto", *ignore_warnings_args],
        }[opt_lvl]

        extra_link_args = {
            "profile" : ["-fopenmp", "-lm", *ignore_warnings_args],
            "0" : ["-fopenmp", "-lm", *ignore_warnings_args],
            "1" : ["-fopenmp", "-lm", *ignore_warnings_args],
            "2" : ["-fopenmp", "-lm", *ignore_warnings_args],
            "3" : ["-fopenmp", "-lm", *ignore_warnings_args],
            "fast" : ["-fopenmp", "-lm", "-flto", *ignore_warnings_args],
        }[opt_lvl]

        # cython_extnames.append("choreo.cython.funs_parallel")
        # cython_safemath_needed.append(False)

else:

    raise ValueError(f"Unsupported platform: {platform.system()}")

cython_filenames = [ ext_name.replace('.','/') + src_ext for ext_name in cython_extnames]

# Special rule for the optional run dependency PyFFTW, disabled in pyodide
cython_extnames.append("choreo.cython.optional_pyfftw")
cython_safemath_needed.append(False)

include_pyfftw = PYFFTW_AVAILABLE and not("PYODIDE" in os.environ) and not(platform.system() == "Windows")

cython_filenames.append(f"choreo.cython.optional_pyfftw_{include_pyfftw}".replace('.','/') + src_ext)

optional_pyfftw_pxd_path = "choreo/cython/optional_pyfftw.pxd"
if include_pyfftw:
    pyfftw_pxd_str = """
cimport pyfftw
import pyfftw as p_pyfftw   
    """
else:
    pyfftw_pxd_str = """
cimport choreo.cython.pyfftw_fake as pyfftw
import choreo.cython.pyfftw_fake as p_pyfftw  
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

# if platform.system() == "Windows":
#     define_macros.append(("CYTHON_CCOMPLEX", 0))

compiler_directives = {
    'wraparound': False         ,
    'boundscheck': False        ,
    'nonecheck': False          ,
    'initializedcheck': False   ,
    'overflowcheck': False      ,
    'overflowcheck.fold': False ,
    'infer_types': True         ,
}

if opt_lvl == "profile" : 

    profile_compiler_directives = {
        'profile': True     ,
        'linetrace': True   ,
        'binding': True     ,
    }
    compiler_directives.update(profile_compiler_directives)
    profile_define_macros = [
        ('CYTHON_TRACE', '1')       ,
        ('CYTHON_TRACE_NOGIL', '1') ,
    ]
    define_macros.extend(profile_define_macros)

include_dirs = [
    numpy.get_include()                 ,
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
    for (name,source,safemath_needed) in zip(cython_extnames,cython_filenames,cython_safemath_needed, strict = True)
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

GUI_data = ['*.js','*.html','assets/**','img/**','python_scripts/**','python_dist/*.whl']
    
if "PYODIDE" in os.environ: # GUI stuff not needed in Pyodide whl
    exclude_package_data['choreo.GUI'].extend(GUI_data)
    
else: # If Pyodide wheel was generated, then copy it!
    
    basedir = os.path.dirname(__file__)
    distdir = os.path.join(basedir, "dist")
    
    if os.path.isdir(distdir):
        
        for filename in os.listdir(os.path.join(basedir, "dist")):
            basename, ext = os.path.splitext(filename)
            
            if 'pyodide' in basename and ext == '.whl':
                
                src = os.path.join(basedir, "dist", filename)
                dst = os.path.join(basedir, "choreo", "GUI", "python_dist", filename)
                
                if os.path.isfile(src):
                        
                    if os.path.isfile(dst):
                        os.remove(dst)
                    
                    shutil.copyfile(src, dst)
    
    package_data['choreo.GUI'].extend(GUI_data)

setuptools.setup(
    ext_modules = ext_modules                   ,
    zip_safe = False                            ,
    packages = packages                         ,
    package_data = package_data                 ,
    exclude_package_data = exclude_package_data ,
    provides = ['choreo']                       ,
)
