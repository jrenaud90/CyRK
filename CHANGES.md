# CyRK - Change Log

## 2023

#### v0.8.4 (TBD)

Performance:
- Removed the try/finally blocks, they were largely not needed as excepts are not implemented so there was no real benefit and they regressed performance.
- Cleaned up used variables in cython-based solvers.

Bug Fixes:
- Max number of steps was being performed before extra_output was parsed in `CySolver` which could lead to incorrect max num steps.
- Fixed incorrect type for `CySolver.user_provided_max_num_steps`.

#### v0.8.3 (2023-10-05)

New Features:
- Added tests to check if memory access violations can occur when `CySolver` is resolved many times.
- Added new cdef methods to `CySolver` for more efficient changes of parameters:
  - `CySolver.change_y0_pointer` - Changes the y0 pointer without having to pass memoryviews.
  - `CySolver.change_t_eval_pointer` - Changes the t_eval pointer without having to pass memoryviews.
  - Added way for user to limit RAM usage for cython-bases solvers. This also changed how max number of steps was calculated.

Performance:
- Changing RK variables back to stack-allocated c-arrays rather than malloc arrays.
- Improved how `CySolver` and `cyrk_ode` expected size is calculated and how much it grows with each concat.
- Files now compile across multiple threads during installation.

Other Changes:
- Moved some common constants for both `CySolver` and `cyrk_ode` out of their files and into `cy.common`.
- Added more meaningful memory error messages to cython files.
  - Memory allocations (or reallocations) are now performed by helper functions in CyRK.utils.
- Better future-proofed package structure (mainifests, gitignores, etc.).
- Converted most Py_ssize_t to size_t.
- Cleaned up a lot of unused variables and imports.

Bug Fixes:
- Fixed potential memory leaks in cython-based solvers when exceptions are raised.
  - The new safe guards (likely the try/finally blocks) did cause a somewhat sizable hit to performance.

#### v0.8.2 (2023-09-25)

New Features:
- Added a helper flag to control if `CySolver.reset_state` is called at the end of initialization.

#### v0.8.1

New Features:
- Added more interp functions that take pointers as arguments.

Changes:
- Converted interp functions to use each other where possible, rather than having separate definitions.
- Cleaned up .pxd file formatting.

Performance:
- Moved some backend functionality for CyRK.interp to pure c file for performance gains.

Bug Fixes:
- Fixed issue with "cy/common.pyx" not having the correct cython flags during compilation.

### v0.8.0

New Features:
- Added new interp functions that work with c pointers. These can only be cimported.
- Added new "CyRK.cy.common.pyx" file for functions that are used by both `cyrk_ode` and `CySolver`.
  - Moved interpolation functionality into `CyRK.cy.common`. Restructured `cyrk_ode` and `CySolver` to use this new function for interpolations.

Changes:
- Refactored many `CySolver` internal attributes to reflect to change from memoryviews to pointers. The most important ones for the user are:
  - `CySolver.y_new_view` -> `CySolver.y_ptr`
  - `CySolver.dy_new_view` -> `CySolver.dy_ptr`
  - `CySolver.t_new` -> `CySolver.t_now`
  - `CySolver.arg_array_view` -> `CySolver.args_ptr`
- Changed RK constants back to c arrays initialized with PyMem_Malloc. The memory for these arrays are setup in the cython-based solvers. Afterwards, there are helper functions in `CyRK.rk` to populate the arrays with correct values.
- Moved to a more generalized scheme for compiling cython files. See "cython_extensions.json", "_build_cyrk.py", and "setup.py" for details.

Performance:
- Transitioned many arrays from numpy to c arrays allocated with PyMem_Malloc, etc. These changes led to a significant performance boost for cython-based solvers.
- Copied some performance lessons that were learned from the cython-based solvers to the numba-based nbrk_ode.

#### v0.7.1

Changes
- Changed cyrk_ode to match the format used by CySolver for its rk_step.

Performance
- Minor calculation taken out of tight inner loops in cython-based solvers. 

Bug Fixes
- Added back noexcepts to dabs functions used by cyrk_ode that were mistakenly removed in final dev commit of v0.7.0.
- Fixed issue where cython-based solvers could overshoot t_span[1].
- Fixed issue where interp functions would give wrong result when requested x was between x_array[0] and x_array[1].
- Fixed issue where interp functions would give wrong results if requested x was negative and x_array was positive (or vice versa).
- The use of carrays for RK constants led to floating point rounding differences that could impact results when step sizes are small.
  - Converted RK constants to numpy arrays which seem to handle the floats much better. 
  - Also changed the interaction with these variables to be done solely through constant memoryviews. This may provide a bit of a performance boost.

### v0.7.0

Major Changes
- Added support for Cython 3.0.0
  - Added `noexcept` to pure cython functions to avoid a potential python error check.

New Features
- Added the ability to pass arrayed versions of rtol and atol to both the numba and cython-based solvers (cyrk_ode and CySolver).
  - For both solvers, you can pass the optional argument "rtols" and/or "atols". These must be C-contiguous numpy arrays with float64 dtypes. They must have the same size as y0.
  - Added tests to check functionality for all solvers.
  - This resolves Issue #1.
- Added new optional argument to all solvers `max_num_steps` which allows the user to control how many steps the solver is allowed to take.
  - If exceeded the integration with fail (softly). 
  - Defaults to 95% of `sys.maxsize` (depends on system architecture).
- New `CySolver.update_constants` method allows for significant speed boosts for certain differential equations.
  - See test diffeqs, which have been updated to use this feature, for examples.

Other Changes
- Improved documentation for most functions and classes.
- To make more logical sense with the wording, `CySolver.size_growths` now gives one less than the solver's growths attribute.
- Cleaned up status codes and created new status code description document under "Documentation/Status and Error Codes.md"
- Fixed compile warning related to NPY_NO_DEPRECATED_API.
- Converted RK variable lengths to Py_ssize_t types.
- Changed default tolerances to match scipy: rtol=1.0e-3, atol=1.0e-6.

Performance
- Various minor performance gains for cython-based solvers.
- Moved key loops in `CySolver` into self-contained method so that gil can be released.
- New `CySolver.update_constants` method allows for significant speed boosts for certain differential equations.

Bug Fixes:
- Fixed potential seg fault when accessing `CySolver`'s arg_array_view.
- Fixed potential issue where `CySolver`'s first step size may not be reset when variables that affect it are.
- Fixed missed declaration in `cyrk_ode`.
- Fixed bug where the state reset flag was not being passed from `CySolver.solve` wrapper method.

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
