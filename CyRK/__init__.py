# Find Version Number
from ._version import version
__version__ = version

# Import numba solver
from .nb.nbrk import nbrk_ode

# Import cython solver
from CyRK.cy.cyrk import cyrk_ode
