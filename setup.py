import os
import platform

import numpy as np
from setuptools import Extension, setup

install_platform = platform.system()

if install_platform.lower() == 'windows':
    extra_compile_args = ['/openmp']
    extra_link_args = []
elif install_platform.lower() == 'darwin':
    extra_compile_args = []
    extra_link_args = []
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

# Cython extensions require a setup.py in addition to pyproject.toml in order to create platform-specific wheels.
setup(
    ext_modules=[
        Extension(
            name='CyRK.array.interp',
            sources=['CyRK/array/interp.pyx'],
            include_dirs=[os.path.join('CyRK', 'array'), np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args),
        Extension(
                name='CyRK.rk.rk',
                sources=['CyRK/rk/rk.pyx'],
                include_dirs=[os.path.join('CyRK', 'rk'), np.get_include()],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args),
        Extension(
            name='CyRK.cy.cyrk',
            sources=['CyRK/cy/_cyrk.pyx'],
            include_dirs=[os.path.join('CyRK', 'cy'), np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    ]
)
