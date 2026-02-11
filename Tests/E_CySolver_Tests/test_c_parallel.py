import os
cpu_count = os.cpu_count()

from CyRK.cy.prange_test import run_prange_common_args_test, run_prange_test

FAIL_ON_BAD_PERFORMANCE = True

# No reason to fail if we only have one logical processor to work with. We wouldn't expect a performance boost.
FAIL_ON_BAD_PERFORMANCE = FAIL_ON_BAD_PERFORMANCE and (cpu_count > 1)

def test_cysolver_prange():
    """Tests that CySolver works in a parallel environment via Cython's prange."""

    # The compiled cython function automatically performance accuracy checks. So we only need to call it
    #  and compare performance if we want.
    threads_1_time = run_prange_test(1)  # When thread count == 0 or 1 it will just use a regular loop
    if cpu_count > 1:
        threads_2_time = run_prange_test(2)
        if FAIL_ON_BAD_PERFORMANCE and (threads_2_time >= threads_1_time):
            raise Exception("CySolver prange test showed that using two threads is slower than one. Unexpected given the test model.")
        
def test_cysolver_prange_common_args():
    """Tests that CySolver works in a parallel environment via Cython's prange.
    
    This function uses common arguments across all threads.
    """

    # The compiled cython function automatically performance accuracy checks. So we only need to call it
    #  and compare performance if we want.
    threads_1_time = run_prange_common_args_test(1)  # When thread count == 0 or 1 it will just use a regular loop
    if cpu_count > 1:
        threads_2_time = run_prange_common_args_test(2)
        if FAIL_ON_BAD_PERFORMANCE and (threads_2_time >= threads_1_time):
            raise Exception("CySolver prange test `run_prange_common_args_test` showed that using two threads is slower than one. Unexpected given the test model.")
