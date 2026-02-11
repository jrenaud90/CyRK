
import platform
import numpy as np
import CyRK

def build_hack(ipython_parser):
    def patched_cython(self, line, cell):
        # Add platform-correct C++ standard flag if not already present
        if "/std:c++" not in line and "-std=c++" not in line:
            if platform.system().lower() == 'windows':
                line += f" --cplus -c /std:c++20"
            else:
                line += f" --cplus -c -std=c++20"

        lines = list()
        line_ = ''
        for char_ in cell:
            if char_ == '\n':
                lines.append(line_)
                line_ = ''
            else:
                line_ += char_

        # Hack the cell to add a distutils parameters to the front of each cell
        lines_prepend = list()
        includes = CyRK.get_include() + [np.get_include()]

        # Add additional include directories to each cell.
        lines_prepend.append(f"# distutils: include_dirs = {includes}")
        if platform.system().lower() == 'windows':
            lines_prepend.append(f'# distutils: extra_compile_args = ["/std:c++20", "/openmp"]')
        else:
            lines_prepend.append(f'# distutils: extra_compile_args = ["-std=c++20", "-O3", "-fopenmp"]')
            if platform.system().lower() == 'darwin':
                lines_prepend.append(f'# distutils: extra_link_args = ["-O3", "-lomp"]')
            else:
                lines_prepend.append(f'# distutils: extra_link_args = ["-O3", "-fopenmp"]')
        lines = lines_prepend + lines
        new_cell = ''
        for line_ in lines:
            new_cell += line_ + '\n'
        return ipython_parser(self, line, new_cell)
    # Preserve the magic parser attribute so IPython still knows how to parse
    patched_cython.parser = ipython_parser.parser
    return patched_cython
