""" Lets the `%%cython` Jupyter magic compile cells against CyRK's C++ headers.

Wrapping IPython's cython magic with `build_hack` (see the demo notebooks) prepends the `# distutils:` directives
every cell needs: CyRK's and NumPy's include directories, the C++20 flag, and, where the compiler supports it, OpenMP
so that `prange` loops run in parallel.

CyRK itself is built without OpenMP; only the module that contains a `prange` loop needs it. MSVC and GCC support
OpenMP out of the box. Apple's clang needs Homebrew's `libomp` (`brew install libomp`); if it is not found the cells
still compile and `prange` loops simply run serially. Set CYRK_DEMO_NO_OPENMP to skip the OpenMP flags everywhere.
"""
import os
import platform

import numpy as np
import CyRK

INSTALL_PLATFORM = platform.system().lower()
HOMEBREW_LIBOMP_PREFIXES = ('/opt/homebrew/opt/libomp', '/usr/local/opt/libomp')


def openmp_flags():
    """ Return the (compile_args, link_args) that enable OpenMP for this platform; both empty if unavailable. """
    if os.environ.get('CYRK_DEMO_NO_OPENMP'):
        return [], []
    if INSTALL_PLATFORM == 'windows':
        return ['/openmp'], []
    if INSTALL_PLATFORM == 'darwin':
        # Apple's clang only understands OpenMP through the preprocessor flag plus an external libomp.
        for prefix in HOMEBREW_LIBOMP_PREFIXES:
            if os.path.isdir(prefix):
                return ['-Xpreprocessor', '-fopenmp', f'-I{prefix}/include'], [f'-L{prefix}/lib', '-lomp']
        print('Homebrew libomp was not found; `prange` loops in %%cython cells will run serially. '
              'Run `brew install libomp` to enable OpenMP.')
        return [], []
    return ['-fopenmp'], ['-fopenmp']


def build_hack(ipython_parser):
    """ Wrap IPython's cython magic so every cell is compiled with CyRK's includes, C++20, and OpenMP if available. """
    openmp_compile_args, openmp_link_args = openmp_flags()
    if INSTALL_PLATFORM == 'windows':
        cpp_standard_flag = '/std:c++20'
        compile_args = [cpp_standard_flag] + openmp_compile_args
        link_args = openmp_link_args
    else:
        cpp_standard_flag = '-std=c++20'
        compile_args = [cpp_standard_flag, '-O3'] + openmp_compile_args
        link_args = ['-O3'] + openmp_link_args
    includes = CyRK.get_include() + [np.get_include()]
    directives = [
        f'# distutils: include_dirs = {includes}',
        f'# distutils: extra_compile_args = {compile_args}',
        f'# distutils: extra_link_args = {link_args}',
        ]

    def patched_cython(self, line, cell):
        # Add the platform's C++ standard flag to the magic's arguments if the cell did not set one.
        if '/std:c++' not in line and '-std=c++' not in line:
            line += f' --cplus -c {cpp_standard_flag}'
        # Prepend the distutils directives to the cell body.
        new_cell = '\n'.join(directives + cell.split('\n')) + '\n'
        return ipython_parser(self, line, new_cell)

    # Preserve the magic parser attribute so IPython still knows how to parse.
    patched_cython.parser = ipython_parser.parser
    return patched_cython
