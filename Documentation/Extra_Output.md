# Capturing Extra Output with CyRK's ODE Solvers

It can be advantageous to capture additional output parameters during the integration of systems of ordinary differential equations.
Particularly when the differential equations in question are computationally expensive. For example, consider the following simplified differential equation.

```python

def diffeq(t, y, a, b):
    
    # Some complicated function of time and y_1; this may be calling many other inner functions.
    f = calculate_f(t, y[1])
    
    # Calculate dy/dt
    dy_0 = f * a * y[0] + b * y[1]
    dy_1 = b * y[1] 
    
    return np.asarray([dy_0, dy_1])
```

Using CyRK it is easy to integrate this function to find $y_0$ and $y_1$ as a function of time.
But suppose that in our final analysis we also want to see how $f$ is changing with time and with $y$'s. We could extract it by using `y_0` and `dy_0` (which we would have to calculate), 
but the more complicated the relationship the harder this is to do and will require additional computations and code.
An easier solution would be to make separate calls to `calculate_f` with the $y$ and $t$ found from integration, but again this requires additional computations which may be expensive.
In this case `calculate_f` will be called _twice_ for each value in the time domain (once for the initial integration and once to find the value of `f` afterwards). 

Instead, it would be ideal if we could just store the value of `f` during integration.

CyRK offers this functionality at the expense of a small hit on performance when using this feature.
The values of these extra parameters are not analyzed by the solver when determining the adaptive step size or numerical error handling.

## Limitations

Current limitations of this feature as of v0.17.0:
- Only numerical parameters can be used as extra outputs (no strings, booleans, other structs).
- All extra outputs must have the same _type_ as the input `y`s. (for `pysolve_ivp` and `cysolve_ivp` extra outputs can only be double floating point numbers).

## How to use with `CyRK.nbsolve_ivp` (Numba-based)

Define a differential equation function with the additional extra outputs put _after_ the y-derivatives.
It is a good idea to turn the output into numpy ndarray before the final return.

```python
import numpy as np
from numba import njit

@njit
def diffeq(t, y, arg_0, arg_1):
    
    extra_parameter_0 = arg_0 * y[0] * np.sin(arg_1 * t)
    extra_parameter_1 = arg_0 * y[0] * np.cos(arg_1 * t - y[1])
    dy_0 = extra_parameter_0
    dy_1 = np.acos(extra_parameter_1)
    
    return np.asarray([dy_0, dy_1, extra_parameter_0, extra_parameter_1], dtype=y.dtype)
```

Call the `nbsolve_ivp` solver in the standard form except now with the `capture_extra` flag to `True` (by default it is `False`).

```python
from CyRK import nbsolve_ivp
time_domain, all_output, success, message = \
    nbsolve_ivp(diffeq, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol, capture_extra=True)
```

The output of the solver (if successful) will now contain both the dependent `y` values over time as well as the extra parameters.

```python
y_0 = all_output[0, :]
y_1 = all_output[1, :]
extra_parameter_0 = all_output[2, :]
extra_parameter_1 = all_output[3, :]
```

## How to use with `CyRK.pysolve_ivp`

The process for capturing extra outputs with the `pysolve_ivp` solver follows similar steps to the numba-based approach where extra outputs are stored in the dy/dt output array.
Note that the extra outputs must be stored at the end of the output array, after all dy/dts.

If you are using a returned dy/dt the format would look like:

```python
def diffeq_cy(t, y, arg_0, arg_1):
    
    dy_size = y.size + 2 # 2 extra spots for the extra output
    dy = np.empty(dy_size, dtype=np.float64)
    extra_0 = (1. - arg_0 * y[1])
    extra_1 = (arg_1 * y[0] - 1.)
    # The differential equations
    dy[0] = extra_0 * y[0]
    dy[1] = extra_1 * y[1]

    # The extra output
    dy[2] = extra_0
    dy[3] = extra_1
```

Or if you pass in the differential equation as the first argument of the diffeq (recommended for improved performance!):

```python
def diffeq_cy(dy, t, y, arg_0, arg_1):
    
    extra_0 = (1. - arg_0 * y[1])
    extra_1 = (arg_1 * y[0] - 1.)
    # The differential equations
    dy[0] = extra_0 * y[0]
    dy[1] = extra_1 * y[1]

    # The extra output
    dy[2] = extra_0
    dy[3] = extra_1
```

The next change is that the solver needs to know how many extra outputs it should expect. In our example there are two additional outputs.
`num_extra` defaults to 0.

```python
from CyRK import pysolve_ivp
result = \
    cyrk_ode(diffeq_cy, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol, num_extra=2)
```

The output matches the output described for the numba-based approach,

```python
y_0 = result.y[0, :]
y_1 = result.y[1, :]
extra_parameter_0 = result.y[2, :]
extra_parameter_1 = result.y[3, :]
```

## Interpolating Extra Outputs

### Interpolating for `cysolve_ivp` and `pysolve_ivp`

When using either `dense_output` or `t_eval`, interpolations must be performed. For dependent $y$ variables, CyRK will utilize the user-specified differential equation method's approach to interpolation. However, this only works for _dependent_ variables.
If you are capturing extra output they will not be included in this interpolation. Instead, both `cysolve_ivp` and `pysolve_ivp` will perform said dependent-variable interpolation and then use the results to make additional calls (one per interpolation, _i.e._, once per value in `t_eval`) to the differential equation to find the values of the extra outputs at the requested time steps.

This approach is similar to "option 1" discussed in the next section for the numba-based solver.

:::{note}
This means that collecting extra outputs when interpolation is one will result in additional calls to the ODE's diffeq. If it is a
computationally expensive function then this could cause a noticeable impact on performance. Read more about managing the
performance of `CyRK` solvers [here](Performance.md).
:::

### Interpolating for numba-based `nbsolve_ivp`

By setting the `t_eval` argument for either the `nbsolve_ivp` solver, an interpolation will occur at the end of integration.
As of CyRK 0.17.0, `nbsolve_ivp` does not support dense output interpolators like the kind discussed in the previous section.
Instead it must rely on (less accurate and less efficient) general purpose, linear interpolators.
These use the solved $y$'s and $t$ to find a new reduced `y_reduced` at `t_eval` intervals using a method similar to `numpy.interp` function. 
Since we are potentially storing extra parameters during integration, we need to tell the solver how to handle any potential interpolation on these additional parameters.

- **Option 1**: Solve for the extra parameters at the new interpolated `y` values
  - _How to set_: Set `t_eval` to the desired numpy array, set `capture_extra=True`, and `interpolate_extra=False`.
  - Pros:
    - Less information is stored during integration which can reduce memory usage, especially when an integration requires many steps.
    - Results are more accurate since all the interpolation error is only in $y$, extra parameters are as correct as $y$ is (no compounding interpolation error).
  - Cons:
    - Additional calls to the `diffeq` are made after integration is completed (number of additional calls `= len(t_eval)`). If this function is computationally expensive then this may have a noticeable performance hit.
- **Option 2**: Store extra parameters during integration and then perform interpolation on both `y` and all extra parameters.
  - _How to set_: Set `t_eval` to the desired numpy array, set `capture_extra=True`, and `interpolate_extra=True`.
  - Pros:
    - No additional calls to `diffeq` are made. This could improve performance if this is a particularly expensive function.
  - Mixed:
    - The same amount of information is stored during integration as if `t_eval` were not set in the first place. This can have a negative impact on memory and performance depending on the number of extra parameters and the number of integration steps.
  - Cons:
    - Results will be less accurate because the extra parameters are interpolated _independent_ of `y`'s interpolation. Parameters that depend on $y$ may no longer match expectations, particularly when $y$ is changing quickly. For example, if an extra output is defined as `x = y[0] * 2.0`, then at some time $t$ where `y[0]` was interpolated to be 5.0, $x$ may differ from its definition (we expect it to be equal to 10.0): `x(t) = 9.8`.
    - More call(s) to `interp` will be made (once for each extra parameter) this may have a negative impact on performance, particularly for many extra parameters.

:::{tip}
It is almost always better to use `interpolate_extra=False`. The results will be more accurate and even if there is a
hit to performance, it should be minimal. A future release of `CyRK` will wrap the `cysolve_ivp` method to work with 
numba, eliminating these considerations.
:::

## Extra Outputs with Events
Extra outputs work with [events](Events) as well. Events store y-data as an array each time an event is triggered. 
If extra outputs are used, then they will be stored with with each dependent y-value for each event that is triggered.
