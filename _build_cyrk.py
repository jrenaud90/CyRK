""" Commands to build the cython extensions of CyRK (a hack to work with pyproject.toml) """
import os
import platform
import math
import json
import sys
from setuptools.extension import Extension
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext

import numpy as np
import Cython

num_procs = os.cpu_count()
num_threads = max(1, num_procs - 1)

install_platform = platform.system()

if install_platform.lower() == 'windows':
    extra_compile_args = ['/openmp']
    extra_link_args = []
elif install_platform.lower() == 'darwin':
    # OpenMP is installed via llvm. See https://stackoverflow.com/questions/60005176/how-to-deal-with-clang-error-unsupported-option-fopenmp-on-travis
    extra_compile_args = ['-O3', '-fopenmp']
    extra_link_args = ['-lomp']
else:
    extra_compile_args = ['-fopenmp', '-O3']
    extra_link_args = ['-fopenmp', '-O3']
macro_list = [("NPY_NO_DEPRECATED_API", "NPY_1_9_API_VERSION")]

# Load CyRK's cython extensions
absolute_path = os.path.dirname(__file__)
cython_ext_path = os.path.join(absolute_path, 'cython_extensions.json')
with open(cython_ext_path, 'r') as cython_ext_file:
    cython_ext_dict = json.load(cython_ext_file)

cython_extensions = list()
for cython_ext, ext_data in cython_ext_dict.items():

    if ext_data['is_cpp']:
        if install_platform.lower() == 'windows':
            specific_compile_args = extra_compile_args + ext_data['compile_args'] + ["/std:c++20"]
        else:
            specific_compile_args = extra_compile_args + ext_data['compile_args'] + ["-std=c++20"]
    else:
        specific_compile_args = extra_compile_args + ext_data['compile_args']

    cython_extensions.append(
        Extension(
            name=ext_data['name'],
            sources=[os.path.join(*tuple(source_path)) for source_path in ext_data['sources']],
            # Always add numpy to any includes; also add sys.path so we can capture python.h
            include_dirs=[os.path.join(*tuple(dir_path)) for dir_path in ext_data['include_dirs']] + [np.get_include()] + sys.path,
            extra_compile_args=specific_compile_args,
            define_macros=macro_list,
            extra_link_args=ext_data['link_args'] + extra_link_args,
            )
        )

class build_ext(_build_ext):

    def run(self):
        # Compile in parallel
        self.parallel = num_threads
        return super().run()

class build_cyrk(_build_py):

    def run(self):
        # Run build_ext to compile cythonized c/cpp files.
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        from Cython.Build import cythonize
        print(f'!-- Cythonizing CyRK (Python v{sys.version}; NumPy v{np.__version__}; Cython v{Cython.__version__})')
        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

        # Add cython extensions to ext_modules list
        for extension in cython_extensions:
            self.distribution.ext_modules.append(
                    extension
                    )

        # Cythonize ext_modules
        self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                compiler_directives={'language_level': "3"},
                include_path=['.', np.get_include()],
                nthreads=num_threads,
                )
        print('!-- Finished Cythonizing CyRK')
