import pytest
import numpy as np
import numba as nb
from CyRK import pysolve_ivp
import math

# define base system
y_0 = 1
t_span = (0.0, 8.0)
@nb.njit("void(f8[::1], f8, f8[::1])")
def cy_diffeq(dy, t, y):
    for i in range(y.size):
        dy[i] = -y[i]/2 + 2*math.sin(3*t)
    

@pytest.mark.parametrize('Neqs', (2**22,))
def test_pysolve_large_neqs(Neqs):
    """ Tests CyRK's ability to simultaneously solve many copies of this system. """
    y0 = np.full(Neqs, y_0, dtype=np.float64)
    # For large Neq we need to increase the max_ram_MB limit.
    result = pysolve_ivp(cy_diffeq, t_span, y0, method="RK45", rtol=1e-7, atol=1e-7, pass_dy_as_arg=True, max_ram_MB=8_000)
    assert result.success
