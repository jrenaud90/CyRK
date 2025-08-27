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

#### Example Usage

### `cysolve_ivp` Version

#### Example Usage

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

