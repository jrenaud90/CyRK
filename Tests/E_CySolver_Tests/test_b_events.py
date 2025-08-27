import numpy as np
import pytest


def test_build_events():
    """Basic tests if the C++ event wrapper class, further wrapped by cython, can be built and initialized correctly."""
    from CyRK.cy.events_test import build_event_wrapper_test

    assert build_event_wrapper_test()

@pytest.mark.parametrize('use_dense', (True, False))
@pytest.mark.parametrize('use_t_eval', (True, False))
def test_run_cysolver_with_events(use_dense, use_t_eval):
    """Test if the solver runs correctly with events."""
    from CyRK.cy.events_test import run_cysolver_with_events

    if use_t_eval:
        t_eval = np.linspace(0, 10, 11, dtype=np.float64)
    else:
        t_eval = np.array([], dtype=np.float64)

    # Run cysolver with events
    assert run_cysolver_with_events(use_dense, t_eval)


if __name__ == "__main__":
    for i in range(100_000):
        print(i)
        test_run_cysolver_with_events(False, False)