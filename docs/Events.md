# Events
Integration "events" are points in time or points in state where the user may want something recorded or want the entire
integration to terminate. 

CyRK largely follows the implementation used by [SciPy's solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

## Event Functions
### `pysolve_ivp` Version
An event is tracked using a user-provided function. The function must take the form of `event(t, y(t), *args)`. "args"
are the same additional arguments provided to `CyRK.pysolve_ivp`. The event then must return a float. Following SciPy's
approach: "\[CyRK\] will find an accurate value of t at which `event(t, y(t), *args) = 0` using a root-finding
algorithm. By default, all zeros will be found. The solver looks for a sign change over each step, so if multiple
zero crossings occur within one step, events may be missed.
Additionally each event function might have the following attributes:"

```
`terminal`: bool or int, optional
    When boolean, whether to terminate integration if this event occurs. When integral, termination occurs after the specified the number of occurrences of this event. Implicitly False if not assigned.
`direction`: float, optional
    Direction of a zero crossing. If direction is positive, event will only trigger when going from negative to positive, and vice versa if direction is negative. If 0, then either direction will trigger event. Implicitly 0 if not assigned.
```
_The above is copied out of SciPy's documentation [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp)._

#### Example
```python
import numpy as np
from CyRK import pysolve_ivp
import matplotlib.pyplot as plt
from numba import njit

# Example with SciPy
@njit
def event_func_1(t, y, a, b, c):

    # Check if t greater than or equal to 5.0
    if t >= 5.0:
        return 0.0
    else:
        return 1.0

@njit
def event_func_2(t, y, a, b, c):

    # Check y values.
    # In the time span [0,10]: 
    #    y_0  starts at 1, spikes then goes below zero and oscillates with a min below -10. Have this return if y_0 < -10
    if y[0] < -10.0:
        return 0.0
    elif y[2] > 30.0:
        return 0.0
    else:
        return 1.0

@njit
def event_func_3(t, y, a, b, c):

    # We won't actually use the args but lets just check they are correct.
    args_correct = False
    if a == 10.0 and b == 28.0 and c == 8.0 / 3.0:
        args_correct = True

    # Then return events if args are correct and t greater than 8
    if args_correct:
        return 0.0
    else:
        return 1.0

@njit
def lorenz_diffeq(dy, t, y, a, b, c):
    # Unpack y
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    
    dy[0] = a * (y1 - y0)
    dy[1] = y0 * (b - y2) - y1
    dy[2] = y0 * y1 - c * y2

def run_pysove_with_events(terminate = False):

    event_func_1.direction = 0
    event_func_1.terminal = 10000
    if terminate:
        event_func_1.terminal = 1
    event_func_2.direction = 0
    event_func_2.terminal = 10000
    event_func_3.direction = 0
    event_func_3.terminal = 10000


    time_span = (0.0, 10.0)
    y0 = np.asarray([1.0, 0.0, 0.0])
    args = (10.0, 28.0, 8.0/3.0)
    solution = pysolve_ivp(lorenz_diffeq, time_span, y0, method="RK45",
                           args=args, rtol=1.0e-6, atol=1.0e-6, t_eval=None, dense_output=False, pass_dy_as_arg=True,
                           events=(event_func_1, event_func_2, event_func_3))

    return solution
```

## Differences from SciPy's Implementation
CyRK allows you to use extra parameters that are captured during integration (see [Extra Output](Extra_Output) for more details) in your user provided event functions. 
They are appended to the `y` array passed to the event function after the dependent variables.


CyRK prefers to output y values as the first index of `y_events` compared to SciPy outputting them as the second.
```python
scipy_solution.y_events == cyrk_solution.y_events.T
```



## Additional Arguments to Event Functions
### `pysolve_ivp` Example:
```python
def event_func(t, y, *args):
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    # Only 3 dependent ys in this example.
    extra0 = y[3]
    extra1 = y[4]
    extra2 = y[5]
    # These extras can now be used to trigger an event!
```

### `cysolve_ivp` Example:
```cython
cdef event_func(double t, double* y, char* args):
    cdef double y0 = y[0]
    cdef double y1 = y[1]
    cdef double y2 = y[2]
    # Only 3 dependent ys in this example.
    cdef double extra0 = y[3]
    cdef double extra1 = y[4]
    cdef double extra2 = y[5]
    # These extras can now be used to trigger an event!
```


## Event Data and Diagnostics in CySolution
CyRK provides several event-related data that can be used to diagnose issues or just learn about the integration results.

```python
solution = pysolve_ivp(... events=(my_event_fuc1, my_event_func2), ...)

# Number of event functions provided, in this case 2.
solution.num_events

# Boolean flag if the integration was terminated by an event.
solution.event_terminated

# If the integration was terminated, then the index of the function the first triggered the termination is provided by the following:
solution.event_terminate_index

# All of this information (plus a lot more!) will be printed if you print the solution diagnostics
solution.print_diagnostics()
```

## Citation
Use of CyRK's integrators with "events" on should cite [SciPy](https://scipy.org/citing-scipy/) since this implementation is based on their work.

