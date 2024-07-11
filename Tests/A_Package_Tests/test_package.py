def test_package():
    """Check if all the functions can be imported. """

    from CyRK import nbsolve_ivp, cyrk_ode, nb2cy, cy2nb, test_nbrk, test_cysolver, test_pysolver, version, __version__, pysolve_ivp

    assert type(version) == str
    assert version == __version__
