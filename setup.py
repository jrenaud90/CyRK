import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

# Find CyRK Directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Find long description
long_description = ''
with open(os.path.join(dir_path, "README.md"), 'r') as f:
    long_description = f.read()

# Find version number
version = ''
with open(os.path.join(dir_path, "version.txt"), 'r') as f:
    for line in f:
        if 'version =' in line:
            version = line.split('=')[-1].strip()

# Dependencies
install_requires = [
    'numpy>=1.1,<1.23.0',
    'numba>=0.55'
]

# Find Cython files and turn them into c code. Must have numpy installed in order to find its c headers.
ext_modules = [Extension('CyRK.cyrk', [os.path.join('CyRK', '_cyrk.pyx')])]

class BuildExtCmd(build_ext):
    def run(self):
        import numpy as np
        self.include_dirs.append(np.get_include())
        super().run()

    def finalize_options(self):
        from Cython.Build import cythonize
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules,
                                                  compiler_directives={'language_level' : "3"})
        super().finalize_options()

setup(
    name='CyRK',
    version=version,
    description='Runge-Kutta ODE Integrator Implemented in Cython and Numba.',
    author='Joe P. Renaud',
    author_email='joe.p.renaud@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jrenaud90/CyRK',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'numpy>=1.1,<1.23.0',
        'cython>=0.29',
    ],
    ext_modules=ext_modules,
    install_requires=install_requires,
    cmdclass={"build_ext": BuildExtCmd},
    packages=find_packages()
)
