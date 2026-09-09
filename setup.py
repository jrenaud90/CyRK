""" Build script for CyRK's Cython extensions.

Package metadata lives in "pyproject.toml". Setuptools can only take extension modules from a "setup.py", and it
decides whether a wheel is platform specific from the extensions handed to `setup()`, so they are declared here. The
extension list itself is kept in "cython_extensions.json" so it can be read without importing this file.

Build-time switches (environment variables):
    CYRK_NO_AVX2: build the x86-64 extensions without AVX2/FMA. "CyRK/_cpu_check.py" reads the same variable.
"""
import os
import sys
import json
import platform

import numpy as np
import Cython
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

DEBUG_MODE = False

# ======================================================================================================================
# Compiler and Linker Flags
# ======================================================================================================================
install_platform = platform.system().lower()

# CyRK's x86-64 builds target AVX2. Set CYRK_NO_AVX2 to build for a CPU that does not have it; the run time check in
# "CyRK/_cpu_check.py" reads the same variable, so leaving it set keeps that check out of the way as well.
USE_AVX2 = not os.environ.get('CYRK_NO_AVX2')

if install_platform == 'windows':
    # 4551 (function call missing argument list) and 4018 (signed/unsigned mismatch) are only ever raised inside the
    # C++ that Cython generates. Silencing them keeps the build log readable so that a warning in CyRK's own sources
    # is actually visible.
    extra_compile_args = ['/O2i', '/GL', '/wd4551', '/wd4018']
    extra_link_args = ['/LTCG']
    if USE_AVX2:
        extra_compile_args.append('/arch:AVX2')
    if DEBUG_MODE:
        # Debug builds turn optimizations off (/Od).
        extra_compile_args = ['/Zi', '/Od']
        extra_link_args = ['/DEBUG:FULL']
    cpp_standard_flag = '/std:c++20'
else:
    extra_compile_args = ['-O3', '-flto']
    extra_link_args = ['-flto']
    if install_platform != 'darwin' and USE_AVX2:
        # Apple silicon has no AVX2 and NEON is already part of its baseline. These must be compile args; passed to
        # the linker they do nothing.
        extra_compile_args.append('-mavx2')
        extra_compile_args.append('-mfma')
    cpp_standard_flag = '-std=c++20'

macro_list = [('NPY_NO_DEPRECATED_API', 'NPY_1_9_API_VERSION')]

# ======================================================================================================================
# Extension Modules
# ======================================================================================================================
setup_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(setup_dir, 'cython_extensions.json'), 'r') as cython_ext_file:
    cython_ext_dict = json.load(cython_ext_file)

# Every extension can see the headers in these sub-packages.
global_include_dirs = [
    os.path.join('CyRK', 'nb'),
    os.path.join('CyRK', 'cy'),
    os.path.join('CyRK', 'utils'),
    os.path.join('CyRK', 'optimize'),
    os.path.join('CyRK', 'array'),
    ]

cython_extensions = list()
for ext_data in cython_ext_dict.values():
    specific_compile_args = extra_compile_args + ext_data['compile_args']
    if ext_data['is_cpp']:
        specific_compile_args.append(cpp_standard_flag)

    cython_extensions.append(
        Extension(
            name=ext_data['name'],
            sources=[os.path.join(*source_path) for source_path in ext_data['sources']],
            include_dirs=(
                global_include_dirs
                + [os.path.join(*dir_path) for dir_path in ext_data['include_dirs']]
                + [np.get_include()]
                ),
            extra_compile_args=specific_compile_args,
            define_macros=macro_list,
            extra_link_args=ext_data['link_args'] + extra_link_args,
            )
        )

# ======================================================================================================================
# Build Command
# ======================================================================================================================
num_threads = 1 if DEBUG_MODE else max(1, (os.cpu_count() or 2) - 1)


class build_ext(_build_ext):
    """ Cythonizes the extensions right before they are compiled, then compiles them in parallel.

    Cythonizing here rather than at `setup()` time keeps commands that only inspect the extensions (`egg_info`,
    `sdist`) from paying for a full Cython pass, while the un-cythonized extensions handed to `setup()` still mark
    the wheel as platform specific.
    """

    def run(self):
        print(f'!-- Cythonizing CyRK (Python v{sys.version}; NumPy v{np.__version__}; Cython v{Cython.__version__})')
        cythonized_extensions = cythonize(
            self.extensions,
            compiler_directives={'language_level': '3'},
            include_path=['.', np.get_include()],
            nthreads=num_threads,
            emit_linenums=DEBUG_MODE,
            )
        if len(cythonized_extensions) != len(self.extensions):
            raise RuntimeError('Cython returned a different number of extensions than it was given.')

        # `cythonize` returns new Extension objects (with the .pyx sources swapped for .cpp and any `# distutils:`
        # directives applied). Setuptools has already annotated the original objects in `finalize_options`, so
        # copy the results onto them instead of replacing them.
        for extension, cythonized_extension in zip(self.extensions, cythonized_extensions):
            for attribute, value in vars(cythonized_extension).items():
                if not attribute.startswith('_'):
                    setattr(extension, attribute, value)
        print('!-- Finished Cythonizing CyRK')

        # Compile the extensions in parallel unless the caller asked for a specific worker count.
        if not self.parallel:
            self.parallel = num_threads
        super().run()


setup(
    ext_modules=cython_extensions,
    cmdclass={'build_ext': build_ext},
    )
