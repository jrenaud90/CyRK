# Find Version Number
import importlib.metadata
__version__ = importlib.metadata.version("CyRK")
version = __version__

# Import numba solver
from .nb.nbrk import nbrk_ode

# Import cython solver
from CyRK.cy.cyrk import cyrk_ode

# Import test functions
from ._test import test_cyrk, test_nbrk
