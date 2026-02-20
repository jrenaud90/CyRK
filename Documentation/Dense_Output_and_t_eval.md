# Dense Output and `t_eval`

*See examples of both these features in the "Demos" folder.*

As of CyRK v0.10.0, support for dense output and accurate interpolation is implemented for all integrators when using `pysolve_ivp` and `cysolve_ivp`.
Note, currently this feature is not available for `nbsolve_ivp` if that is a something you would like to see in the future please open a new issue on GitHub.

## `t_eval`

Since CyRK uses an adaptive time stepping scheme it will only record results at time steps that have acceptable error (governed by the user-provided tolerances and the integration method).
The resulting time domain (and associated y-values) will be non-uniform and may be more or less densely spaced than the user may wish.
To correct this, a user may provide an array via the `t_eval` argument. CyRK will then compute its adaptive steps as usual, but will only save data for each `t_eval` point.
If a `t_eval` point falls in-between the adaptive steps, then CyRK will use an interpolator (see next section) to determine a well-informed guess for the value.

Notes:
- `t_eval` must be within the provided `time_span`.

## Dense Output

In addition to the time domain, y-values, and any extra output (see "Extra Output.md"), CyRK can also build and store localized interpolator functions.
These are constructed at each successful time step using the current state of the problems derivatives, y-values, and time information. 
The interpolators can then be called with a desired time `t` and the resulting `y` will be produced, even if that specific `t` was never hit during integration (as long at `t` is within the provided `time_span`). 
These interpolators are what is used for estimating y values when `t_eval` is provided.

Notes:
- The dense outputs are relatively large and must be heap allocated at each time step. Therefore it is rather computationally expensive to store them. Leave `dense_output=False` unless required to improve performance.
    - This performance hit is much less noticeable if only `t_eval` is provided because in that case we can utilize a single stack allocation. 

## Interpolating Extra Outputs with Dense Output

As discussed in the "Extra Output.md" documentation, CyRK can capture additional outputs from the differential equation process. These are non-dependent variables (non-y values) that are not used during integration error calculations but may be useful data for the user.
This is triggered when `num_extra` is set to > 0 in either `pysolve_ivp` or `cysolve_ivp`. If `dense_output` is also set to True, then the final solution interpolators will also interpolate these extra outputs. This is done by making additional calls to the differential equation to determine what the values of the extra outputs are at each interpolated time step. 
More details can be found in "Extra Output.md"

### Using Dense Outputs

Example:
```python
def cy_diffeq(dy, t, y):
    dy[0] = (1. - 0.01 * y[1]) * y[0]
    dy[1] = (0.02 * y[0] - 1.) * y[1]

    # If using extra output: set `num_extra=2`
    # dy[2] = (1. - 0.01 * y[1])
    # dy[3] = (0.02 * y[0] - 1.) 

import numpy as np
from CyRK import pysolve_ivp

initial_conds = np.asarray((20., 20.), dtype=np.complex128, order='C')
time_span = (0., 50.)
rtol = 1.0e-7
atol = 1.0e-8

result = \
    pysolve_ivp(cy_diffeq, time_span, initial_conds, method="RK45", dense_output=True,
                rtol=rtol, atol=atol,
                pass_dy_as_arg=True)
print("Was Integration was successful?", result.success)
print(result.message)
print("Size of solution: ", result.size)
import matplotlib.pyplot as plt
fig0, ax0 = plt.subplots()
ax0.plot(result.t, result.y[0], c='r')
ax0.plot(result.t, result.y[1], c='b')

# Call the result function to use the dense output interpolators.
new_time_array = np.linspace(5.0, 10.0, 250)
interpolated_y = result(new_time_array)

fig1, ax1 = plt.subplots()
ax1.plot(new_time_array, interpolated_y[0], c='r')
ax1.plot(new_time_array, interpolated_y[1], c='b')
```
