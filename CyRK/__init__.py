# Find Version Number
import importlib.metadata
__version__ = importlib.metadata.version("CyRK")
version = __version__

# Import numba solver
from .nb.nbrk import nbsolve_ivp

# Import helper functions
from .helper import nb2cy, cy2nb

# Import test functions
from ._test import test_nbrk, test_cysolver, test_pysolver

# Import python solver
from CyRK.cy.cysolver_api import WrapCySolverResult
from CyRK.cy.pysolver import pysolve_ivp

# Helper function that provides directories to CyRK c++ headers
def get_include():
    import os
    import CyRK
    cyrk_dir = os.path.dirname(CyRK.__file__)

    cyrk_dirs = list()
    cyrk_dirs.append(
        os.path.join(cyrk_dir, 'cy')  # CySolver headers
    )
    cyrk_dirs.append(
        os.path.join(cyrk_dir, 'array')  # Array headers
    )

    return cyrk_dirs
