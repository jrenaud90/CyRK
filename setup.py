from setuptools import setup, find_packages
from setuptools.extension import Extension

from version import version
ext = Extension('cyrk', sources=[os.path.join('CyRK', 'cyrk.pyx')])

# Import Cython
try:
    from Cython.Build import cythonize
except:
    raise ImportError('Cython not found and is required to build CyRK.')
    
# Import numpy
try:
    import numpy as np
except:
    raise ImportError('Numpy not found and is required to build CyRK.')

# Dependencies
install_requires = [
    'numpy>=1.23'
]

# Find Cython files and turn them into c code. Must have numpy installed in order to find its c headers.
ext_modules = cythonize(
    [
        Extension('CyRK', [os.path.join('CyRK', 'cyrk.pyx')], include_dirs=[np.get_include()])
    ]
)

setup(
    name='CyRK',
    version=version,
    description='Runge-Kutta ODE Integrator Implemented in Cython and Numba.',
    author='Joe P. Renaud',
    author_email='joe.p.renaud@gmail.com',
    url='https://github.com/jrenaud90/CyRK',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'numpy>=1.23',
        'cython>=0.29',
    ],
    ext_modules=ext_modules,
    install_requires=install_requires
)
