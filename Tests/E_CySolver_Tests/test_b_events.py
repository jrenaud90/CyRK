import numpy as np
import pytest


def test_build_events():
    """Basic tests if the C++ event wrapper class, further wrapped by cython, can be built and initialized correctly."""
    from CyRK.cy.events_test import build_event_wrapper_test

    assert build_event_wrapper_test()

@pytest.mark.parametrize('use_dense', (True, False))
@pytest.mark.parametrize('use_t_eval', (True, False))
@pytest.mark.parametrize('use_termination', (True, False))
@pytest.mark.parametrize('use_capture_extra', (True, False))
def test_run_cysolver_with_events(use_dense, use_t_eval, use_termination, use_capture_extra):
    """Test if the solver runs correctly with events."""
    from CyRK.cy.events_test import run_cysolver_with_events

    if use_t_eval:
        t_eval = np.linspace(0, 10, 11, dtype=np.float64)
    else:
        t_eval = np.array([], dtype=np.float64)

    # Run cysolver with events
    solution, cython_tests_passed = run_cysolver_with_events(use_dense, t_eval, use_termination, use_capture_extra)
    assert cython_tests_passed

    assert type(solution.t_events) == list
    assert len(solution.t_events) == 3
    assert type(solution.y_events) == list
    assert len(solution.y_events) == 3
    assert solution.event_terminated == use_termination
    if use_termination:
        assert solution.event_terminate_index == 0
    
    # Loop through 3 events
    for i in range(3):
        if i == 0 and use_termination:
            assert solution.t_events[i].size == 1

        if use_capture_extra:
            assert solution.y_events[i].shape[0] == 6  # 3 dependent y values + 3 extras
        else:
            assert solution.y_events[i].shape[0] == 3  # 3 dependent y values
        assert solution.t_events[i].size == solution.y_events[i][0].size

        if use_capture_extra:
            # The first extra output (4th output including dependent ys) is always a constant in this example
            assert np.all(solution.y_events[i][3] == 10.0)
    
    if use_termination:
        assert solution.t[-1] < 5.1


if __name__ == "__main__":
    for i in range(100_000):
        print(i)
        test_run_cysolver_with_events(False, False, True, True)