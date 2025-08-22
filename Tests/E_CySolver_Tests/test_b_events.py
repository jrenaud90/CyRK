import numpy as np
import pytest


def test_build_events():
    """Basic tests if the C++ event wrapper class, further wrapped by cython, can be built and initialized correctly."""
    from CyRK.cy.events_test import build_event_wrapper_test

    assert build_event_wrapper_test()
