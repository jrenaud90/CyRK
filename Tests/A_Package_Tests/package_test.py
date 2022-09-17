

def test_package():
    """Check if all the functions can be imported. """

    from CyRK import cyrk_ode, nbrk_ode, version

    assert type(version) == str
