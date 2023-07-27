""" Commands to build the cython extensions of CyRK (a hack to work with pyproject.toml) """
import os
import platform
from setuptools.extension import Extension
from setuptools.command.build_py import build_py as _build_py

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

class build_cyrk(_build_py):

    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        import numpy as np
        from Cython.Build import cythonize
        print('!-- Cythonizing CyRK')
        if self.distribution.ext_modules == None:
            self.distribution.ext_modules = []

        # Add array to ext_modules list
        self.distribution.ext_modules.append(
                Extension(
                        name='CyRK.array.interp',
                        sources=[os.path.join('CyRK', 'array', 'interp.pyx')],
                        include_dirs=[os.path.join('CyRK', 'array'), np.get_include()],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args
                        )
                )

        # Add RK constants to ext_modules list
        self.distribution.ext_modules.append(
                Extension(
                        name='CyRK.rk.rk',
                        sources=[os.path.join('CyRK', 'rk', 'rk.pyx')],
                        include_dirs=[os.path.join('CyRK', 'rk'), np.get_include()],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args
                        )
                )

        # Add cyrk to ext_modules list
        self.distribution.ext_modules.append(
                Extension(
                        'CyRK.cy.cyrk',
                        sources=[os.path.join('CyRK', 'cy', 'cyrk.pyx')],
                        include_dirs=[os.path.join('CyRK', 'cy'), np.get_include()],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args
                        )
                )
        
        # Add CySolver to ext_modules list
        self.distribution.ext_modules.append(
                Extension(
                        'CyRK.cy.cysolver',
                        sources=[os.path.join('CyRK', 'cy', 'cysolver.pyx')],
                        include_dirs=[os.path.join('CyRK', 'cy'), np.get_include()],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args
                        )
                )
        
        # Add CySolverTest to ext_modules list
        self.distribution.ext_modules.append(
                Extension(
                        'CyRK.cy.cysolvertest',
                        sources=[os.path.join('CyRK', 'cy', 'cysolvertest.pyx')],
                        include_dirs=[os.path.join('CyRK', 'cy'), np.get_include()],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args
                        )
                )

        # Add cythonize ext_modules
        self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                compiler_directives={'language_level': "3"},
                include_path=['.', np.get_include()]
                )
        print('!-- Finished Cythonizing CyRK')
