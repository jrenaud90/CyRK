""" Commands to build the cython extensions of CyRK (a hack to work with pyproject.toml) """
import os

from setuptools.extension import Extension

from setuptools.command.build_py import build_py as _build_py


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

        # Add cyrk to ext_modules list
        self.distribution.ext_modules.append(
                Extension(
                    'CyRK.cy.cyrk',
                    sources=[os.path.join('CyRK', 'cy', '_cyrk.pyx')],
                    include_dirs=[os.path.join('CyRK', 'cy'), np.get_include()]
                    )
                )

        # Add cythonize ext_modules
        self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                compiler_directives={'language_level': "3"},
                include_path = ['.', np.get_include()]
                )
        print('!-- Finished Cythonizing CyRK')
