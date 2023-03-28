""" Commands to build the cython extensions of CyRK (a hack to work with pyproject.toml) """
import os
from setuptools.extension import Extension
from setuptools.command.build_py import build_py as _build_py

is_windows = False
if os.name == 'nt':
    is_windows = True

if is_windows:
    extra_compile_args = ['/openmp']
    extra_link_args = ['/openmp']
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
                        sources=['CyRK/array/interp.pyx'],
                        include_dirs=[os.path.join('CyRK', 'array'), np.get_include()],
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args
                        )
                )

        # Add cyrk to ext_modules list
        self.distribution.ext_modules.append(
                Extension(
                        'CyRK.cy.cyrk',
                        sources=[os.path.join('CyRK', 'cy', '_cyrk.pyx')],
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
