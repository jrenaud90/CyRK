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

Using CyRK it is easy to integrate this function to find `y_0` and `y_1` as a function of time.
But suppose that in our final analysis we also want to see how `f` is changing with time and `y`s. We could extract it by using `y_0` and `dy_0` (which we would have to calculate), 
but the more complicated the relationship the harder this is to do and will require additional computations and code.
An easier solution would be to make separate calls to `calculate_f` for our new `y` and `t`, but again this requires additional computations which may be expensive.
In this case `calculate_f` will be called _twice_ for each value in the time domain (once for the initial integration and once to find the value of `f` afterwards). 

Instead, it would be ideal if we could just store the value of `f` during integration since it is already being calculated during the solving process.

CyRK offers this functionality at the expense of a small hit on performance when using this feature.
The values of these extra parameters are not analyzed by the solver when determining the adaptive step size or numerical error handling.

## Limitations

Current limitations of this feature as of v0.4.0:
- Only numerical parameters can be used as extra outputs (no strings or booleans).
- All extra outputs must have the same _type_ as the input `y`s. This means if you are using `y`s which are floats, but you need to capture a complex number, 
then you will either need to change the `y`s to complex-valued (with a zero imaginary portion) or, a better option, return the real and imaginary portions of the extra parameter separately.

## How to use with `CyRK.nbrk_ode`

Define a differential equation function with the additional extra outputs put _after_ the y-derivatives.
It is a good idea to turn the output into numpy ndarray before the final return.

```python
import numpy as np

def diffeq(t, y, arg_0, arg_1):
    
    extra_parameter_0 = arg_0 * y[0] * np.sin(arg_1 * t)
    extra_parameter_1 = arg_0 * y[0] * np.cos(arg_1 * t - y[1])
    dy_0 = extra_parameter_0
    dy_1 = np.acos(extra_parameter_1)
    
    return np.asarray([dy_0, dy_1, extra_parameter_0, extra_parameter_1], dtype=y.dtype)
```

Call the `nbrk_ode` solver in the standard form except now with the `capture_extra` flag to `True` (by default it is `False`).

```python
from CyRK import nbrk_ode
time_domain, all_output, success, message = \
    nbrk_ode(diffeq, time_span, initial_conds, rk_method=1, rtol=rtol, atol=atol, capture_extra=True)
```

The output of the solver (if successful) will now contain both the dependent `y` values over time as well as the extra parameters.

```python
y_0 = all_output[0, :]
y_1 = all_output[1, :]
extra_parameter_0 = all_output[2, :]
extra_parameter_1 = all_output[3, :]
```

## Extra considerations when using `t_eval`

By setting the `t_eval` argument for either the `nbrk_ode` or `cyrk_ode` solver, an interpolation will occur at the end of integration.
This uses the solved `y`s and `time_domain` to find a new reduced `y_reduced` at `t_eval` intervals using `numpy.interp` function. 
Since we are potentially storing extra parameters during integration, we need to tell the solvers how to handle any potential interpolation on these new parameters.

- **Option 1**: Solve for the extra parameters at the new interpolated `y` values
  - _How to set_: Set `t_eval` to the desired numpy array, set `capture_extra=True`, and `interpolate_extra=False`.
  - Pros:
    - Less information is stored during integration which can reduce memory usage for when an integration requires many steps.
    - Results are more accurate since all the interpolation error is in `y`, extra parameters are correct (with respect to the new `y`s) and not interpolated further.
  - Cons:
    - Additional calls to the `diffeq` are made after integration is completed (number of calls `= len(t_eval)`). If this function is computationally expensive then this will slow down overall solver time.
- **Option 2**: Store extra parameters during integration and then perform interpolation on both `y` and all extra parameters.
  - _How to set_: Set `t_eval` to the desired numpy array, set `capture_extra=True`, and `interpolate_extra=True`.
  - Pros:
    - No additional calls to `diffeq` are made. This could improve performance if this is a particularly expensive function.
  - Mixed:
    - The same amount of information is stored during integration as if `t_eval` were not set. This can have a negative impact on memory and performance depending on the number of extra parameters.
  - Cons:
    - Results are less accurate because the extra parameters are interpolated independent of `y` so if any of the parameters depend on `y` then the interpolated results may no longer match expectations. Particularly when `y` is changing quickly.
    - More call(s) to `numpy.interp` will be made (once for each extra parameter) this may have a negative impact on performance, particularly for many extra parameters.
