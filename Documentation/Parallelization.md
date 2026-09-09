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

### When is threading worth it?
Two costs are added by `c_cysolve_batch`: starting and joining the worker threads (a fixed cost per batch), and, for
very short jobs, contention between threads for memory allocation and bandwidth. The table below was measured with
`CyRK.cy.parallel_test.run_parallel_benchmark` on a Windows 11 desktop (MSVC build, 16 logical CPUs, Python 3.13);
each row is the best of three runs of the same Lotka-Volterra problem, with `t_end` chosen to set the time per job.
Starting and joining the threads cost 0.23 ms for two threads, 0.27 ms for four, and 0.46 ms for eight.

| Time per job | Jobs | Serial batch | 2 threads | 4 threads | 8 threads |
|-------------:|-----:|-------------:|----------:|----------:|----------:|
| 1.6 us       | 4000 | 6.6 ms       | 1.8x      | 3.0x      | 4.0x      |
| 10 us        | 4000 | 42 ms        | 2.0x      | 3.7x      | 5.8x      |
| 100 us       | 2000 | 0.20 s       | 1.8x      | 3.5x      | 6.7x      |
| 1.2 ms       | 1000 | 1.2 s        | 2.1x      | 4.0x      | 7.7x      |
| 12 ms        | 200  | 2.3 s        | 2.0x      | 3.9x      | 7.6x      |
| 120 ms       | 32   | 3.9 s        | 2.0x      | 4.0x      | 7.8x      |
| 1.3 s        | 4    | 5.2 s        | 2.0x      | 4.0x      | 4.0x (only 4 jobs) |

Rules of thumb drawn from it (Linux and macOS start threads more cheaply than Windows, so they lean further toward
threading):

* The fixed cost is roughly 0.1 ms per thread started. Threading pays once the serial batch takes at least ten times
  that, so **time per job x number of jobs of about 5 ms or more**. A 10 us problem needs a batch of a few hundred
  jobs; a 100 us problem a few dozen; a 1 ms problem only a handful.
* Below about 10 us per job, expect around 2x from two threads but only 4x to 6x from eight; the jobs spend a large
  share of their time in the solver's setup and allocations, which threads compete for. From about 100 us per job the
  speedup is close to the thread count, and from 1 ms up it is essentially linear.
* A job is never split, so the speedup can not exceed the number of jobs, and the batch ends when the slowest thread
  finishes its block. Give each thread several jobs of similar length; if the jobs differ a lot in cost, interleave
  the long and short ones or run more, smaller batches.
* Numbers differ between machines. Run `run_parallel_benchmark(num_threads, num_runs, t_end)` for a few values of
  `t_end` on the target machine to find its own crossover.
