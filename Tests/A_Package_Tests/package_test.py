def test_package():
    """Check if all the functions can be imported. """

    from CyRK import cyrk_ode, nbrk_ode, version, __version__, test_cyrk, test_nbrk

    assert type(version) == str
    assert version == __version__
