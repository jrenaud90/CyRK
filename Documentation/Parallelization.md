# Parallelizing CyRK

:::{note}
The discussion on this page only pertains to CyRK's `cysolve_ivp`, `pysolve_ivp`, and related methods.
`nbsolve_ivp` and `nbsolve2_ivp` may support parallelization but it has not been tested thoroughly and is not
officially supported at this time.
:::

The inner workings of CySolver are not parallelized on purpose: generally the performance gains of parallelizing the
integration steps are far out weighed by the complexity, errors, and most importantly, overhead of distributed work.
However, the functions that interact with that backend (`pysolve_ivp`, `cysolve_ivp`, and their derivatives) can
be use in parallelized loops. This can greatly speed up programs that perform many, slow integrations.

## Parallel `pysolve_ivp`
`pysolve_ivp` function can be parallelized using Python's
[multiprocessing](https://docs.python.org/3/library/multiprocessing.html) package.
Note that it can not utilize multithreading because `pysolve_ivp` requires a reference to the user-provided,
Python differential equation. This would be shared across threads leading to inadvertent serialization if not just
crashing. Examples on how this is done can be found in the
[Getting Started notebook](https://cyrk.readthedocs.io/en/latest/Demos/1_-_Getting_Started.html#Parallelizing-pysolve_ivp).

## Parallel `cysolve_ivp`
`cysolve_ivp` and `cysolve_ivp_noreturn` are `nogil` and thread safe as long as each thread works on its own
`CySolverResult`. They can therefore be driven by any threading mechanism: a Cython
[prange](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html) loop, C++ threads (`std::thread`)
inside your own extension, or Python threads calling a `nogil` wrapper.

CyRK's own binaries are built without OpenMP and do not need it. It is the module that contains the `prange` loop that
must be compiled with OpenMP, using the flags for its compiler:

| Compiler            | Compile flags                                  | Link flags               |
|---------------------|------------------------------------------------|--------------------------|
| MSVC (Windows)      | `/openmp`                                      |                          |
| GCC (Linux)         | `-fopenmp`                                     | `-fopenmp`               |
| Apple clang (macOS) | `-Xpreprocessor -fopenmp -I<libomp>/include`   | `-L<libomp>/lib -lomp`   |

On macOS `<libomp>` is Homebrew's `libomp` package (`brew install libomp`, installed under `/opt/homebrew/opt/libomp`
on Apple silicon or `/usr/local/opt/libomp` on Intel). Without OpenMP, Cython compiles `prange` into an ordinary loop:
the code still runs, just serially.

Examples on how this is done can be found in the
[Advanced CySolver Examples notebook](https://cyrk.readthedocs.io/en/latest/Demos/2_-_Advanced_CySolver_Examples.html#Parallelizing-cysolve_ivp)
(its `%%cython` cells get the flags above from `Demos/jupyter_cyhack.py`, which only adds them where the compiler
supports OpenMP).

### C++ threads: `c_cysolve_batch`
CyRK ships a small C++ helper to assist with parallelization. `c_cysolve_batch` takes a vector of `c_CySolveJob`
structs (one `CySolverResult` per job plus the diffeq, time span, initial conditions, and any optional inputs) and
spreads them over `std::thread` workers. Each thread solves a contiguous block of jobs, so no locking is needed.
cimport it from `CyRK.cy.parallel`:

```cython
from libcpp.vector cimport vector
from libcpp.memory cimport make_unique
from libcpp.utility cimport move
from CyRK.cy.cysolver_api cimport CySolverResult, CySolveOutput, ODEMethod
from CyRK.cy.parallel cimport c_CySolveJob, c_cysolve_batch

cdef vector[CySolveOutput] results = vector[CySolveOutput](num_jobs)
cdef vector[c_CySolveJob] jobs
cdef c_CySolveJob job
for i in range(num_jobs):
    results[i] = move(make_unique[CySolverResult](ODEMethod.RK45))
    job = c_CySolveJob()
    job.solution_ptr = results[i].get()
    job.diffeq_ptr   = my_diffeq
    job.t_start      = 0.0
    job.t_end        = 100.0
    job.y0_vec_ptr   = &y0_vecs[i]
    job.args_vec_ptr = &args_vec      # May be shared: the solver copies the vectors it is given.
    jobs.push_back(job)

with nogil:
    c_cysolve_batch(jobs, num_threads)
```

The job struct's optional fields (`rtol`, `atol`, per-variable tolerance vectors, `t_eval_vec_ptr`, `events_vec_ptr`,
`pre_eval_func`, `max_num_steps`, and so on) default to the same values as `cysolve_ivp_noreturn`. A failure inside a
job is recorded on that job's `CySolverResult` (`success`, `status`, `message`) rather than raised. The complete,
runnable version of this example is `CyRK.cy.parallel_test`, which is what the test suite uses to check that the
solver is safe to run from several threads.
