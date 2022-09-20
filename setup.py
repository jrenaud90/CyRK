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
with open(os.path.join(dir_path, "CyRK", "_version.py"), 'r') as f:
    for line in f:
        if 'version =' in line:
            version = line.split('=')[-1].strip().replace("'", '').replace('"', '')

# Dependencies
setup_requirements = [
    # Setuptools 18.0 properly handles Cython extensions.
    'setuptools>=18.0',
    'numpy>=1.20.3,<1.23',
    'cython>=0.29.30'
    ]

requirements = [
    'numba>=0.54.1',
    'numpy>=1.20.3,<1.23'
    ]

# Meta Data
classifiers = [
    "Development Status :: 4 - Beta",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: Implementation :: CPython",
    "Natural Language :: English",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics"
    ]

# Find Cython files and turn them into c code. Must have numpy installed in order to find its c headers.
ext_modules = [
    Extension('CyRK.cy.cyrk',
              sources=[os.path.join('CyRK', 'cy', '_cyrk.pyx')],
              include_dirs=[os.path.join('CyRK', 'cy')]
              )
    ]


# Create a build ext class that waits until setup dependencies are installed before cythonizing and building c-code
class BuildExtCmd(build_ext):

    def run(self):
        import numpy as np
        self.include_dirs.append(np.get_include())
        super().run()

    def finalize_options(self):
        from Cython.Build import cythonize
        print('!-- Cythonizing CyRK')
        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules,
            compiler_directives={'language_level': "3"}
            )
        print('!-- Finished Cythonizing CyRK')
        super().finalize_options()


setup(
    name='CyRK',
    version=version,
    description='Runge-Kutta ODE Integrator Implemented in Cython and Numba.',
    author='Joe P. Renaud',
    author_email='joe.p.renaud@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    url='https://github.com/jrenaud90/CyRK',
    repository="https://github.com/jrenaud90/CyRK/",
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    ext_modules=ext_modules,
    install_requires=requirements,
    cmdclass={"build_ext": BuildExtCmd},
    packages=find_packages(),
    package_data={"CyRK.cy": ["_cyrk.pyx"]},
    )
