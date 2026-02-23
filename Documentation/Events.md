# Events
Integration "events" are points in time or state where the user may want something recorded or have the entire
integration to terminate early. 

CyRK largely follows the implementation used by [SciPy's solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

## Event Functions
### `pysolve_ivp` Version
An event is tracked using a user-provided function. The function must take the form of `event(t, y(t), *args)`. "args"
are the same additional arguments provided to the differential equation function. The event then must return a float. Following SciPy's
approach: "\[CyRK\] will find an accurate value of $t$ at which `event(t, y(t), *args) = 0` using a root-finding
algorithm. By default, all zeros will be found. The solver looks for a sign change over each step, so if multiple
zero crossings occur within one step, events may be missed.
Additionally each event function might have the following attributes:"

```
`event.terminal`: bool or int, optional
    If provided as a bool = True, then it will tell the integrator to stop integration if this
      event is ever 0.
    If an int > 1, then termination occurs after this number of occurrences of this event.
    If not defined then integrator will not terminate if this event is hit (implicitly False).
`event.direction`: float, optional
    Direction of a zero crossing.
    If direction is positive, event will only trigger when going from negative to positive or
      from 0 to 0, and vice versa if direction is negative.
    If 0, then either direction will trigger event.
    Implicitly 0 if not assigned.
```
_The above is largely copied from SciPy's documentation [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp)._

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
    if args_correct and t > 8.0:
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
    if terminate:
        event_func_1.terminal = 1
    event_func_2.direction = 0
    event_func_3.direction = 0


    time_span = (0.0, 10.0)
    y0 = np.asarray([1.0, 0.0, 0.0])
    args = (10.0, 28.0, 8.0/3.0)
    solution = pysolve_ivp(lorenz_diffeq, time_span, y0, method="RK45",
                           args=args, rtol=1.0e-6, atol=1.0e-6, t_eval=None, dense_output=False, pass_dy_as_arg=True,
                           events=(event_func_1, event_func_2, event_func_3))

    return solution
```

### `cysolve_ivp` Implementation
Please see the Advanced CySolver demo notebook to see this in action.
```python
%%cython --force
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libcpp.limits cimport numeric_limits

import numpy as np
cimport numpy as cnp
cnp.import_array()

from CyRK cimport CyrkErrorCodes, PreEvalFunc, MAX_STEP, DiffeqFuncType, Event, EventFunc, cysolve_ivp_noreturn, WrapCySolverResult, ODEMethod, CySolverResult
from CyRK.cy.cysolver_test cimport lorenz_diffeq

cdef double lorenz_event_func_1(double t, double* y_ptr, char* unused_args) noexcept nogil:

    # Check if t greater than or equal to 5.0
    if t >= 5.0:
        return 0.0
    else:
        return 1.0

cdef double lorenz_event_func_2(double t, double* y_ptr, char* unused_args) noexcept nogil:

    # Check y values.
    # In the time span [0,10]: 
    #    y_0  starts at 1, spikes then goes below zero and oscillates with a min below -10. Have this return if y_0 < -10
    if y_ptr[0] < -10.0:
        return 0.0
    elif y_ptr[2] > 30.0:
        return 0.0
    else:
        return 1.0

cdef double lorenz_event_func_3(double t, double* y_ptr, char* args) noexcept nogil:

    # Use arguments to set threshold
    cdef double* args_dbl_ptr = <double*>args

    # We won't actually use the args but lets just check they are correct.
    cdef cpp_bool args_correct = False
    if args_dbl_ptr[0] == 10.0 and args_dbl_ptr[1] == 28.0 and args_dbl_ptr[2] == 8.0 / 3.0:
        args_correct = True

    # Then return events if args are correct and t greater than 4
    if args_correct and t >= 4.0:
        if t <= 4.1:
            return 0.0
        elif t >= 4.5 and t <= 4.6:
            return 0.0
        elif t >= 5.0 and t <= 5.1:
            return 0.0
        elif t >= 5.5 and t <= 5.6:
            return 0.0
        elif t >= 6.0 and t <= 6.1:
            return 0.0
        else:
            return 1.0
    else:
        return 1.0

def run_cysolver_with_events(bint use_termination = False):
    
    # Inputs for lorenz diffeq
    cdef double t_start = 0.0
    cdef double t_end = 10.0
    
    cdef size_t num_y = 3
    cdef vector[double] y0_vec = vector[double](num_y)
    y0_vec[0] = 1.0
    y0_vec[1] = 0.0
    y0_vec[2] = 0.0
    
    cdef vector[double] t_eval_vec = vector[double]()
    cdef vector[char] args_vec = vector[char](3 * sizeof(double))
    args_dbl_ptr = <double*>args_vec.data()
    args_dbl_ptr[0] = 10.0
    args_dbl_ptr[1] = 28.0
    args_dbl_ptr[2] = 8.0 / 3.0
    
    cdef PreEvalFunc pre_eval_func = NULL
    
    # Build events
    cdef vector[Event] events_vec = vector[Event]()
    # Set the maximum number of events allowed before termination to the max size of size_t (effectively infinite)
    cdef size_t max_allowed = numeric_limits[size_t].max()
    cdef int direction = 0
    if use_termination:
        events_vec.emplace_back(lorenz_event_func_1, 1, 0)
    else:
        events_vec.emplace_back(lorenz_event_func_1, max_allowed, direction)
    events_vec.emplace_back(lorenz_event_func_2, max_allowed, direction)
    events_vec.emplace_back(lorenz_event_func_3, max_allowed, direction)
    
    cdef ODEMethod integration_method = ODEMethod.RK45
    cdef WrapCySolverResult solution = WrapCySolverResult()
    solution.build_cyresult(integration_method)
    cdef CySolverResult* solution_ptr = solution.cyresult_uptr.get()
    
    cdef DiffeqFuncType diffeq = lorenz_diffeq
    cdef size_t num_extra = 0
    cdef cpp_bool use_dense = False
    
     # Run Solver
    cysolve_ivp_noreturn(
        solution_ptr,
        diffeq,
        t_start,
        t_end,
        y0_vec,
        rtol = 1.0e-6,
        atol = 1.0e-6,
        args_vec = args_vec,
        num_extra = num_extra,
        max_num_steps = 100000,
        max_ram_MB = 2000,
        dense_output = use_dense,
        t_eval_vec = t_eval_vec,
        pre_eval_func = pre_eval_func,
        events_vec = events_vec,
        rtols_vec = vector[double](),
        atols_vec = vector[double](),
        max_step = MAX_STEP,
        first_step = 0.0,
        expected_size = 128
        )
    
    solution.finalize()
    return solution
```

To plot the data we can use a python cell. 
```python
# Compare to the solve_ivp and pysolve_ivp examples in "1 - Getting Started" Notebook

# CyRK v0.17.0
#    Terminate off: 57.9us; 56.9us; 58.5us
#    Terminate on:  29.4us; 30.6us; 29.6us
terminate = True

solution = run_cysolver_with_events(terminate)
print("Solution Succeeded:", solution.success)

%timeit run_cysolver_with_events(terminate)

fig, ax = plt.subplots()
ax.plot(solution.t, solution.y[0], label='y0', c='C0')
ax.plot(solution.t, solution.y[1], label='y1', c='C3')
ax.plot(solution.t, solution.y[2], label='y2', c='C6')
# Event 1 will be dots.
# Event 2 will be X's
# Event 3 will be +'s
ax.scatter(solution.t_events[1], solution.y_events[1][0, :], c='C0', marker='o')
ax.scatter(solution.t_events[0], solution.y_events[0][1, :], c='C3', marker='x', s=120)
ax.scatter(solution.t_events[1], solution.y_events[1][2, :], c='C6', marker='o')
ax.scatter(solution.t_events[2], solution.y_events[2][0, :], c='C0', marker='+', s=120)
ax.set(xlabel='Time', ylabel='y')
ax.legend(loc="upper right")
plt.show()
```

```{image} ./_static/imgs/events.png
:alt: CyRK Events Example
:width: 600px
:align: center
```

## Differences from SciPy's Implementation
CyRK is capable of capturing extra parameters during integration (see [Extra Output](Extra_Output) for more details). 
These will also be captured during event triggers and can be used in an event to determine if an event should trigger.
They are appended to the `y` array passed to the event function after the dependent variables.


CyRK outputs y values as the first index of `y_events` compared to SciPy outputting them as the second.
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


## Event Diagnostics
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

