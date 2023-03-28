import os

import numpy as np
from setuptools import Extension, setup

is_windows = False
if os.name == 'nt':
    is_windows = True

if is_windows:
    extra_compile_args = ['/openmp']
    extra_link_args = ['/openmp']
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
            name='CyRK.cy.cyrk',
            sources=['CyRK/cy/_cyrk.pyx'],
            include_dirs=[os.path.join('CyRK', 'cy'), np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    ]
)
