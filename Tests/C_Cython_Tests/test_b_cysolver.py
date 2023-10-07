def test_cysolver_test():
    """Check that the builtin test function for the CySolver integrator is working"""

    from CyRK import test_cysolver
    test_cysolver()