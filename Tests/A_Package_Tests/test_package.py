def test_package():
    """Check if all the functions can be imported. """

    from CyRK import nbsolve_ivp, nb2cy, cy2nb, test_nbrk, test_cysolver, test_pysolver, version, __version__, pysolve_ivp

    assert type(version) == str
    assert version == __version__

def test_testers_nbrk():

    from CyRK import test_nbrk

    test_nbrk()

    assert True

def test_testers_cysolver():

    from CyRK import test_cysolver

    test_cysolver()

    assert True

def test_testers_pysolver():

    from CyRK import test_pysolver

    test_pysolver()

    assert True
