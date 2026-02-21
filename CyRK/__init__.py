# Find Version Number
import importlib.metadata
__version__ = importlib.metadata.version("CyRK")
version = __version__

# Import python solver
from CyRK.cy.common import CyrkErrorCodes, MAX_SIZE
from CyRK.cy.cysolver_api import WrapCySolverResult, ODEMethod
from CyRK.cy.pysolver import pysolve_ivp, PySolver
from CyRK.cy.pyhelpers import get_error_message, find_ode_method_int

# Import numba solver
from .nb.nbrk import nbsolve_ivp
from .nb.numba_solver import nbsolve2_ivp

# Import helper functions
from .helper import nb2cy, cy2nb

# Import test functions
from ._test import test_nbrk, test_cysolver, test_pysolver

# Helper function that provides directories to CyRK c++ headers
def get_include():
    import os
    import CyRK
    cyrk_dir = os.path.dirname(CyRK.__file__)

    cyrk_dirs = list()
    cyrk_dirs.append(
        os.path.join(cyrk_dir, 'array')  # Array headers
    )
    cyrk_dirs.append(
        os.path.join(cyrk_dir, 'optimize')  # Array headers
    )
    cyrk_dirs.append(
        os.path.join(cyrk_dir, 'cy')  # CySolver headers
    )
    cyrk_dirs.append(
        os.path.join(cyrk_dir, 'nb')  # Array headers
    )

    return cyrk_dirs
