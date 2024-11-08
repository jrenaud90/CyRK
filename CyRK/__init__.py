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