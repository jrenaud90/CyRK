""" Checks that CySolver can be driven from several C++ threads at once (see CyRK/cy/parallel_test.pyx). """
import os
import warnings

from CyRK.cy.parallel_test import run_parallel_common_args_test, run_parallel_test

cpu_count = os.cpu_count() or 1

# A second thread should be faster on the test model, but shared CI runners make timings noisy, so a slow result only
# warns by default. Turn FAIL_ON_BAD_PERFORMANCE on to make it a failure on a quiet machine with more than one core.
FAIL_ON_BAD_PERFORMANCE = False
WARN_ON_BAD_PERFORMANCE = True
FAIL_ON_BAD_PERFORMANCE = FAIL_ON_BAD_PERFORMANCE and (cpu_count > 1)


def _check_threads(runner, name: str):
    """ Run `runner` on one thread, then on two, and compare the times. The runner checks the results itself. """
    threads_1_time = runner(1)
    if cpu_count > 1:
        threads_2_time = runner(2)
        if threads_2_time >= threads_1_time:
            msg = (f"CySolver parallel test `{name}` showed that using two threads ({threads_2_time:0.3f} ms) is "
                   f"slower than one ({threads_1_time:0.3f} ms). Unexpected given the test model, but it can happen "
                   "on a busy machine.")
            if FAIL_ON_BAD_PERFORMANCE:
                raise Exception(msg)
            elif WARN_ON_BAD_PERFORMANCE:
                warnings.warn(msg, RuntimeWarning)


def test_cysolver_parallel():
    """ Each job has its own argument vector. """
    _check_threads(run_parallel_test, "test_cysolver_parallel")


def test_cysolver_parallel_common_args():
    """ Every job reads the same argument vector, which the solver must treat as read only. """
    _check_threads(run_parallel_common_args_test, "test_cysolver_parallel_common_args")
