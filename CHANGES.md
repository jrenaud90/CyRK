# CyRK - Change Log

## 2025

#### v0.14.0 (2025-07-XX)

Changes:
* C++ Backend:
  * Moved to ErrorCodes enum for tracking error codes.
  * Changed how CySolverResult works:
    * Transitioned away from using shared pointers.
    * Instantiating classes requires less arguments. Moving more of the loading on to resets so that memory does not need
  * Added more checks to look out for memory allocation problems.
  * Added more checks and error catches to avoid other problems.
  * Optimized solution storage, solver, and dense output to better fit in typical cache sizes.
  * Vectors are now pre-allocated with guesstimated sizes at objection creation to reduce the need for subsequent reallocations.
to be reallocated for simple re-runs of the same ODE.

Fixes:
- CySolver Backend:
  - Fixed potential issue where provided t-eval pointer could stop pointing to correct block of memory.
  - Fixed issue where extra outputs were not getting saved on the first time step.
  - Fixed potential problem where t_eval indices were being converted to ints. For very large arrays there could be a wrap around bug.
  - Moved to one vector for dependent y arrays storage and one vector for dy arrays. Pointers added to respective sections of each vector.
  - Most vectors are now resized and values are changed via assignment rather than being cleared and appended to. This allows for resets to retain the original memory and avoid memory allocations unless absolutely necessary. 
- PySolver Backend:
  - Fixed potential issue where the python module may not have gotten decremented. 
  - Removed the PySolver methods from the C++ codebase. Don't see any reason they can't live in pure python land.

#### v0.13.5 (2025-04-09)

Fixes:
* Missing files in manifest.

#### v0.13.4 (2025-04-09)

Fixes:
* Fixed *.pyx and *.pxd files not being included in distributions.

#### v0.13.3 (2025-04-09)

Fixes:
* Fixed issue in manifest.

#### v0.13.2 (2025-04-09)

Changes:
* Modified GitHub workflows so that tests are only run on commits when asked.

Fixes:
* Fixed cythonized files that were being included with sdist, causing problems with conda-forge.

#### v0.13.1 (2025-04-08)

Changes:
* Added more debug info during installation.

Fixes:
* Fixed const errors at compile time.

### v0.13.0 (2025-04-03)

New & Changes:
* Added support for Python 3.13 and Numpy 2.x

Fixes:
* Fixes memory leak in CySolver (this was mostly effecting pure cython calls; python calls may not have had an issue. Also only when dense output was on).
  * The fix also led to a fairly big performance boost to `pysolve`.
* Fixed error in CySolver tester that could cause crashes or incorrect results (only effected some tests and benchmarks).

#### v0.12.2 (2025-03-27)

* Re-release to trigger conda build that included dependency fixes.

## 2024

#### v0.12.1 (2024-12-02)

Fixes:
* Added fix so that arrays returned by `WrapCySolverResult` class are not destroyed when class is. Instead they are managed by python's garbage collector once they are held by a python variable. This addresses Github issues [#80](https://github.com/jrenaud90/CyRK/issues/80) and [#78](https://github.com/jrenaud90/CyRK/issues/78)

Tests:
* Added test to check that arrays stay alive after underlying `WrapCySolverResult` class is destroyed.

#### v0.12.0 (2024-12-02)

New & Changes:
* `MAX_STEP` can now be cimported from the top-level of CyRK: `from CyRK cimport MAX_STEP`
* Changed uses of `void*` to `char*` for both diffeq additional args and pre-eval outputs. The signature of these functions has changed, please review documentation for correct usage.
* Added new diagnostic info display tool to `cysolve_ivp` and `pysolve_ivp` output that you can access with `<result>.print_diagnostics()`.

Fixes:
* Fixed issue with `cysolve_ivp` (`pysolve_ivp` did not have this bug) where additional args are passed to diffeq _and_ dense output is on _and_ extra output is captured.
  * Calling the dense output when extra output is on requires additional calls to the diffeq. However, after integration there is no gurantee that the args pointer is pointing to the same memory or that the values in that memory have not changed.
  * Best case, this could cause unexpected results as new values are used for additional args; worst case it could cause access violations if the diffeq tries to access released memory.
  * Now the `CySolverBase` makes a copy of the additional argument structure so it always retains those values as long as it is alive. This requires an additional memory allocation at the start of integration. And it requires the user to provide the size of the additional argument structure to `cysolve_ivp`.

Tests:
* Fixed tests where additional args were not being used.
* Fixed issue with diffeq test 5.
  * This fixes GitHub Issue [#67](https://github.com/jrenaud90/CyRK/issues/67)
* Added tests to solve large number of diffeqs simultaneously to try to catch issues related to GitHub issue [#78](https://github.com/jrenaud90/CyRK/issues/78).

Documentation:
* Updated the "Advanced CySolver.md" documentation that was out of date.

#### v0.11.5 (2024-11-27)

New:
* Added a `steps_taken` tracking variable to the C++ class `CySolverResult` and the Cython wrapped `WrapCySolverResult` so that users can see how many steps were taken during integration even when using `t_eval`.

#### v0.11.4 (2024-11-25)

C++ Changes
* Moved away from stack to heap allocated vectors for arrays that depend on the size of y0. Prior stack allocation required a hard upper limit which was a limitation for some use cases (See GitHub Issue [#74](https://github.com/jrenaud90/CyRK/issues/74)).

#### v0.11.3 (2024-11-18)

New:
* Added helper function to allow users to include CyRK headers during installation. `CyRK.get_include()` this returns a list which must be appended to your include list in "setup.py".
* Some minor performance improvements when using dense output or t_eval.
* Added new cysolve_ivp C++ function, `baseline_cysolve_ivp_noreturn`, and cython wrapped version, `cysolve_ivp_noreturn`, that require the storage class as input (then modify it) rather than provide it as output.

C++ Changes:
* Major rework of the C++ Solver backend
  * The `CySolverResult` storage structure is now the owner of: the solution data, the dense outputs, _and_ the solver itself (before the solver was stand alone). This ensures that the solver is alive for the dense outputs which, depending on settings, may need to make calls to the solver even after the integration is complete. 
  * Removed many pointers in favor of direct access of variables. This was due to some variables moving and hanging pointers causing crashes.
  * Drastically reduced the size of `CySolverDense` and moved its state data to a heap allocated array. This has led to large performance gains when `dense_output=True` for integrations requiring a large number of steps.
* `num_y` and `num_extra` are now size_t type instead of unsigned ints.
* `integration method` is not int instead of unsigned int.

Fixes:
* Fixed issue where if t_eval was less than t_end it could cause an access violation.
* Fixed issue where dense output was sometimes being created twice when t_eval was provided.
* Addressed several compile warnings.
* Fixed issue where when t_eval is provided and dense output = true, cysolver may call a dense solution that has moved (hanging pointer).

#### v0.11.2 (2024-11-12)

New:
* Using extra output (via `num_extra`) with `cysolve_ivp` and `pysolve_ivp` now works when `dense_output` is set to true. CySolverSolution will now make additional calls to the differential equation to determine correct values for extra outputs which are provided alongside the interpolated y values.
  * Added relevant tests.

Other:
* Refactored some misspellings in the cysolver c++ backend.

Fix:
* Fixed missing `np.import_array` in cysolver backend.

#### v0.11.1 (2024-11-11)

Fixes:
* MacOS was not correctly installing openmp version of the cython object files. Fixed this but...

Issues:
* Cython `prange` was failing (for the array interps) only on MacOS on some versions of Python. Couldn't figure it out so removing prange for now.

### v0.11.0 (2024-11-09)

New:
* `WrapCySolverResult` result class now provides user access to attribute `num_y`.

Removed:
* Removed previous `cyrk_ode` and older version of the `CySolver` class-based solver.
  * The functionality of `cyrk_ode` is now handled by the new (as of v0.10.0) `pysolve_ivp` function.
  * The functionality of `CySolver` is partly handled by the new (as of v0.10.0) `cysolve_ivp` function.
    * Note that the new cysolve_ivp is a functional approach. A class based approach like the older CySolver class supported is no longer available but could be easy to implement. If there is interest please create a Github issue for it.

Refactors:
* Refactored the new cysolver and pysolver files to remove "New". This will break imports based on previous versions.
* Broke up cysolver and pysolver into different files to isolate each other's code.

Other:
* Changed the default ordering for diffeq function inputs to follow the scheme dydt(dy, t, y); previously it was dydt(t, y, dy). This affects the `cy2nb` and `nb2cy` helper functions.
* Updated performance module to use new methods over old.

Demos:
* Fixed typo in the type of the mixed-type args container.
* Updated to work with new refactoring.

Tests:
* Updated tests to use pysolver where cyrk_ode was used.
* Changed tolerances and other inputs to try to make some tests faster.

Dependencies:
* Tested that CyRK works with numpy v2.X; but a lot of other packages don't right now. So setting it as upper limit.
* Tested that CyRK can not work with Python 3.13 yet due to numba dependence. See issue 

#### v0.10.2 (2024-11-05)

New:
* Added new `interpolate_from_solution_list` function to help interpolate between multiple solutions across a domain.

Bugs:
* Fixed issue where `CyRK.test_cysolver()` used incorrect kind and order of arguments.
* Fixed MacOS compile issues when using OpenMP (for both x86-64 and arm64 macos).
* Fixed issue where MacOS was failing certain tests.
* Building new wheels to fix Github issue [#62](https://github.com/jrenaud90/CyRK/issues/62).

Tests:
* Added tests to check all built in testers.

#### v0.10.1 (2024-07-25)

C++ Backend:
* Changed optional args from double pointer to void pointers to allow for arbitrary objects to be passed in.
  * Added description of this feature to "Documentation/Advanced CySolver.md" documentation and "Demos/Advanced CySolver Examples.ipynb" jupyter notebook.
* Allow users to specify a "Pre-Eval" function that can be passed to the differential equation. This function should take in time, y, and args and update an output pointer which can then be used by the diffeq to solve for dydt.

`cysolve_ivp`:
* Change call signature to accept new `pre_eval_func` function.
* Added more differential equations to tests.
* Added tests to check new void arg feature.
* Added tests to check new pre-eval function feature.

MacOS:
* Going back to GCC for C and C++ compile instead of clang (ran into inconsistent test failures with clang).

### v0.10.0 (2024-07-17)

C++ Backend:
* This version of CyRK introduces a major rework of the backend integrator which is now written in pure C++.
* CySolver is now a Cython wrapper to this C++ integrator which can be accessed via Python.
  * Access this function by using `from CyRK cimport cysolve_ivp` (this must be done within Cython).
  * The plan is to replace CyRK's older `CySolver` with this function.
* There is now a new PySolver version of this wrapper that allows a user to pass a python function to the C++ backend.
  * Access this function by using `from CyRK import pysolve_ivp`.
  * This is designed as a drop-in-place replacement for SciPy's `solve_ivp`.
  * The plan is to replace CyRK's older `cyrk_ode` with this function.

Implemented Dense Output and Improved `t_eval` for new C++ backend:
* Both `pysolve_ivp` and `cysolve_ivp` now utilize a much more accurate interpolator when `t_eval` is provided.
* Users can now also request the interpolators be saved with the data, enabling Dense Output functional calls.
* This closes [#45](https://github.com/jrenaud90/CyRK/issues/45).
  * Note that these improvement was not made for `nbsolve_ivp`, `cyrk_ode`, or `CySolver` methods. See below to learn about these methods' deprecation.
* Added tests, documentation, and demos describing these features.

Deprecating Older CyRK Methods:
* The new C++ backend is more flexible, faster, and allows for easy additions of new features. It is common across the cython-only, python, and njit-safe numba solvers. Therefore making a change to it propagates to all three solvers - allowing for easier maintenance and new features. For these reasons, the older `cyrk_ode`, `CySolver`, and `nbrk_ode` are now marked as deprecated. No new features will be implemented for those functions and they will be removed in the next major release of CyRK.
* Deprecated `cyrk_ode`
* Deprecated `CySolver`
* Warnings will be issued if these functions are used in this release. To suppress these warnings set `raise_warnings` to False in the respective function calls.

CySolver:
* Changed error message to use a stack-allocated char-array and associated pointer.
* Added new argument to constructor `raise_warnings` (default: True) to allow users to suppress warnings.

cyrk_ode:
* Added new argument to constructor `raise_warnings` (default: True) to allow users to suppress warnings.

WrapCySolverResult:
* `cysolve_ivp` and `pysolve_ivp` now return a class structure that stores the result of the integration along with some meta data. The accessible attributes are:
  * `cysolve_ivp`s `CySolverResult`:
    * success (bool): Flag if the integration was successful.
    * message (str): Message to give a hint on what happened during integration.
    * error_code (int): Additional error/status code that hints on what happened during integration.
    * size (int): Length of time domain.
    * y (float[:, :]): 2D Float Array of y solutions (and any extra output).
    * t (float[:]): 1D Float Array of time domain at which y is defined.

numba based `nbsolve_ivp`:
* The older `nbrk_ode` has been refactored to `nbsolve_ivp` to match the signature of the new cython-based functions (and scipy's solve_ivp).
* The output of `nbsolve_ivp` is now a named tuple that functions similar to the `WrapCySolverResult`

Memory Management:
* Changed exit code when memory can not be allocated.
* Changed some heap allocated arrays in `CySolver` to be stack allocated
  * This change limits the total number of y-dependent variables and extra output that is tracked to 50. This size is
  easy to change. If your use case requires a larger size then open an issue and an alternative can be discussed.
* Converted the underlying storage arrays for `CySolver` to LinkedLists arrays.

Bug Fixes:
* Fixed issue where the Cython-based solvers might use the incorrect memory freeing function.

Other Changes:
* Moved from GCC to Clang on MacOS builds. There was a new problem that appeared with GCC's linker and could not find a working solution. The original move away from clang was done to support openMP multiprocessing. CyRK does not currently use that so the switch back should be okay.

Known Issues:
* There is an occasional bug with backwards integration on pysolve_ivp and cysolve_ivp. See [Github Issue #56](https://github.com/jrenaud90/CyRK/issues/56).

### v0.9.0 (2024-05-22)

Major Changes:
- Shifted from using the Python-based `PyMem_Alloc`, `PyMem_Free` to c-based `malloc`, `free`. 
- CySolver `_solve` method is now gil-free.
  - This has led to a 35%--230% speed boost at low values of steps (faster start up).

Other Changes:
- CI will now build x64-86 and arm64 wheels for MacOS (change suggested by @dihm in [#49](https://github.com/jrenaud90/CyRK/issues/49)).
  - Did have to use this `nomkl` [workaround](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial) which may cause problems. TBD.

#### v0.8.8 (2024-04-30)

New Features:
* Added `utils.free_mem` function to free memory so that future changes to the memory allocation system will call the proper free function that works with the `utils.allocs`.

Changes:
* Changed all instances of `PyMem_Free` to new `free_mem` from the utils. 

#### v0.8.7 (2024-04-28)

Updated manifest and rebuilt wheels.

#### v0.8.6 (2024-02-13)

Major Changes:
- Added support for Python 3.12.
- Converted `CySolver`'s `rk_step` method into a pure-c implementation to allow for further optimizations.
- Changed all files to compile with c rather than c++.
  - Had to change cpp_bools to bints to make this change.

Bug Fixes:
- Fixed issue where CyRK was not installing on MacOS due to issue with LLVM and OpenMP. 
  - Have opted to go to gcc for macOS install due to issues with OpenMP and clang.
- Fixed incorrect type for rk method in CySolver (should eliminate some compile warnings).
- Fixed issue in benchmark where incorrect results were being displayed for CySolver.

## 2023

#### v0.8.5 (2023-10-27)

Other Changes:
- Improved t-eval setter and resetter in `CySolver`

Bug Fixes:
- Fixed bug where incorrect memory was being accessed whenever `CySolver` integration failed (leading to seg fault).


#### v0.8.4 (2023-10-18)

Performance:
- Removed some try/finally blocks, they were largely not needed in `CySolver` as allocated memory is released on the class destruction.
- Cleaned up unused variables in cython-based solvers.

Other Changes:
- Added "force_fail" parameter to `CySolver` to force the integrator to fail to test if memory is released properly.
- `CySolver` class pointers now initialize to NULL at the start of init.
- `CySolver` now owns all data that is heap-allocated (via class attributes). This allows better management of data in the event of crashes or integration failures.

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
