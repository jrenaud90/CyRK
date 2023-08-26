# CyRK - Change Log

## 2023

### v0.7.0

New Features
- Added new optional argument to all solvers `max_steps` which allows the user to control how many steps the solver is allowed to take.
  - If exceeded the integration with fail (softly). 
  - Defaults to 95% of `sys.maxsize` (depends on system architecture).

Other Changes
- Refactored `max_step` to `max_step_size` argument for all solvers to avoid confusion with new `max_steps` argument. 
- Improved documentation for `CySolver`'s `diffeq` method template.
- To make more logical sense with the wording, `CySolver.size_growths` now gives one less than the solver's growths attribute.
- Cleaned up status codes and created new status code description document under "Documentation/Status and Error Codes.md"
- Fixed compile warning related to NPY_NO_DEPRECATED_API.

Performance
- Various minor performance gains for cython-based solvers.

Bug Fixes:
- Fixed potential seg fault when accessing `CySolver`'s arg_array_view.
- Fixed potential issue where `CySolver`'s first step size may not be reset when variables that affect it are.

#### v0.6.2
New Features
- Added `auto_solve` key word to `CySolver` class. This flag defaults to True. If True, then the solver will automatically call `self.solve()` after initialization.
- Added new parameter change functions to `CySolver` so that certain parameters can be changed after the class is initialized for a performance boost.
  - Look for the "self.change_<X>" methods in cysolver.pyx/pxd. There is a main change method, `CySolver.change_parameters` which allows you to change multiple parameters at once.

Bug Fixes:
- Fixed issue where `CySolver` could give incorrect results if the `solve()` method was called multiple times on the same instance.
- Removed extraneous code from `CySolver.__init__`.
- Changed several cython integer variables to all use Py_ssize_t types. Corrected type conversions.

#### v0.6.1

New Features
- Added top level parameters (like `MAX_STEP`) used in `CySolver` to `cysolver.pxd` so they can be cimported.
- Added new argument to `array.interp` and `array.interp_complex`: `provided_j` the user can provide a `j` index,
allowing the functions to skip the binary search. 
- Added new function `interpj` to array module that outputs the interpolation result as well as the `j` index that was found.

Bug Fixes
- Fixed issue with array tests not actually checking values.

Other Changes
- Reordered tests since numba tests take the longest.
- Added `initializedcheck=False` to the array module compile arguments.

### v0.6.0

New Features
- `CyRK` now works with python 3.11.
- Created the `CySolver` class which is more efficient than the `cyrk_ode` function.
  - Solves [issue 28](https://github.com/jrenaud90/CyRK/issues/28)
- New functions in `CyRK.cy.cysolvertest` to help test and check performance of `CySolver`.

Performance
- Removed python lists from `cyrk_ode` leading to an increase in performance of 15--20%. 
  - Solves [issue 27](https://github.com/jrenaud90/CyRK/issues/27)

Bug Fixes:
- Fixed compile error with `cyrk_ode` "complex types are unordered".
  - This was not a problem before so likely something has changed in newer cython versions.
- Fixed missing declarations for variables in `cyrk_ode`.
- Fixed potential problems during installation where paths may be incorrect depending on OS.

#### v0.5.3

Performance
- Removed dynamic optional arguments from `cyrk_ode`. Now it checks if those arguments are set to None.

Other Changes
- Changed `cyrk_ode` arguments to const to avoid memoryview buffer problems. (Change made by [David Meyer](https://github.com/dihm))

#### v0.5.1 and v0.5.2

- Fixed issues with MacOS wheel build during CI.

### v0.5.0

New Features
- `cyrk_ode` now supports both float and complex-typed y and dydt functions.
  - Resolves [issue 3](https://github.com/jrenaud90/CyRK/issues/3)). (Feature added by [David Meyer](https://github.com/dihm))

Performance
- Converted various ints to `short`s, `char`s, or `Py_ssize_t`. `Py_ssize_t` is recommended by Cython for loop integers to better support 64-bit architecture. 
- Added custom interpolation functions which, depending on the size of the array, can be up to 10x faster than numpys.
- Removed unnecessarily variables from `cyrk_ode`.
- Had to turn off `fastmath` for `nbrk_ode`. See [issue 24](https://github.com/jrenaud90/CyRK/issues/24). This negatively impacted the numba integrator's performance by around 5%.

Other Changes
- Refactored, cleaned up, and added comments and docstrings to `cyrk_ode`.
- Changed both `nbrk_ode` and `cyrk_ode` tests to use pytest parameterization.
- Changed the accuracy test for both `nbrk_ode` and `cyrk_ode` to check against a known function.
- Added openmp dependence during compile time to allow for the use of `prange`.
- Moved `cyrk_ode`'s Runge-Kutta constants to a separate module `CyRK.rk`. 

Bug Fixes:
- Fixed issue (for `nbrk_ode` and `cyrk_ode`) where incorrect step size could be used due to bad minimum step check (see [issue 20](https://github.com/jrenaud90/CyRK/issues/20)).

### v0.4.0

New Features
- Added the ability to save intermediate (non-dependent y) results during integration for `nbrk` and `cyrk` ode solver.
  - See `Documentation/Extra Output.md` for more information. 

Performance
- Minor performance improvements to `cyrk_ode` (switch to c++ compiler and some related functionality)
- The new feature that saves intermediate results during integration had a minor impact on performance (even when not using the feature). However, it is within most tests margin of error.

### v0.3.0

Bug Fixes:
- `nbrk_ode` fixes
  - Improved the storage of results during integration, greatly reducing memory usage. This provides a massive increase in performance when dealing with large time spans that previously required the processor to search outside its cache during integration.
    - This fixes [issue 5](https://github.com/jrenaud90/CyRK/issues/5).

Performance Improvements
- Various improvements to `nbrk_ode` make it about 200% faster on small time-spans and over 30x+ faster on large timespans.
- Improvements to `cyrk_ode` provided a modest (~5%) performance increase.

Other Changes
- Helper functions now have an additional optional kwarg `cache_njit` which is set to `False` but can be toggled to enable njit caching.
- Fixed issue in function timing calculation used in the benchmark plot.

#### v0.2.4

Bug Fixes
- Fixed issue in precompiled wheel distribution ([issue 9](https://github.com/jrenaud90/CyRK/issues/9)). (Fix made by [Caroline Russell](https://github.com/cerrussell))

Other Changes
- Updated CI workflows to utilize `cibuildwheel` for building binary wheels.

## 2022

#### v0.2.3

Bug Fixes
- `cyrk_ode` fixes
  - Bug in doubling up on the time step in the final inter-step diffeq calculation.

#### v0.2.2

Other Changes
- Added a performance tracking package to measure CyRK's performance over time and versions.

#### v0.2.1

New Features
- Added helper functions `from CyRK import nb2cy, cy2nb` which convert differential equation argument signatures between the formats required for cyrk and nbrk ode solvers.

Other Changes
- Added back some commented out tests that were left over from the bug fixed in v0.2.0.
- Added tests to check that performs both cyrk and nbrk integrations on larger time domains.
- Removed the specific test that looked at the underlying issue fixed in v0.2.0 (this is still checked by other tests).

### v0.2.0

Bug Fixes
- Fixed issues with the metadata provided by pyproject.toml.
- `cyrk_ode` fixes
  - Fixed bug that was causing the ubuntu slowdown and likely other errors
  - Added a `cabs` absolute value function to ensure that complex numbers are being properly handled when `abs()` is called.
- `nbrk_ode` fixes
  - Fixed warning during numba integration about contiguous arrays.
  - Fixed issue where variable was referenced before assignment when using nbrk's DOP853

Performance Improvements
- `cyrk_ode` improvements
  - Integrator now selects an output message based on the status code rather than building a string during the integration loop.
  - Switched the loop order for the final list to ndarray conversion. Before the time domain was being redundantly built y_size times.

Other Changes
- pyproject.toml provides more constrained package list rather than an open search.
- Added back ubuntu tests and publishes to GitHub workflows.

### v0.1.3

- Fixed a broken example in the readme documentation.
- Added better quick tests for both the numba and cython versions.
- Added `SciPy` dependency that was missing (required for numba integrator).
- Increased the lower limit on `numpy` package version (to fix issue [7](https://github.com/jrenaud90/CyRK/issues/7)) and removed the upper limit version.
- Removed python 3.7 as a supported version. 
- Updated graphic in readme.
- Converted over to using `pyproject.toml` instead of `setup.py`
  - removed `version.py` from project folder.

### v0.1.2

- Made the calling argument for the numba solver more consistent with the cython one by letting first_step==0 be
equivalent to == None

### v0.1.1

- Corrected issues with installation
- Improved GitHub workflows

### Initial Beta Version 0.1.0 Launched!
