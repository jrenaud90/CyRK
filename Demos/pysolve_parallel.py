import numpy as np
from CyRK import pysolve_ivp

# Define your ODE (Top-level, picklable)
def my_diffeq(t, y, a, b, c, d):
    # Standard Lotka-Volterra: dx/dt = ax - bxy
    return np.array([a * y[0] - b * y[0] * y[1], 
                     -c * y[1] + d * y[0] * y[1]])

# Define the worker function here
def solve_worker(params):
    # Unpack arguments
    diffeq, t_span, y0, args, rtol, atol = params

    # Run solver
    result = pysolve_ivp(
        py_diffeq=diffeq,
        time_span=t_span,
        y0=y0,
        args=args,
        rtol=rtol,
        atol=atol,
        method='RK45'
    )

    # Return pure data
    return {
        't': result.t,
        'y': result.y,
        'success': result.success,
        'status_message': result.status_message
    }
