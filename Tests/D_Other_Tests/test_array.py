import numpy as np

def test_array_module():
    """ Test that the functions load correctly. """
    from CyRK.array import interp, interp_complex, interp_array, interp_complex_array

def test_interp_array():
    """ Test custom interpolation function (float arrays). """
    from CyRK.array import interp_array

    t = np.linspace(0., 1000., 500, dtype=np.float64)
    t_small = np.linspace(0., 1000., 50, dtype=np.float64)
    x = 100 * np.cos(t)**2

    numpy_interp = np.interp(t_small, t, x)

    cyrk_interp = np.empty(t_small.size, dtype=np.float64)
    interp_array(t_small, t, x, cyrk_interp)

    np.allclose(numpy_interp, cyrk_interp)

def test_interp_complex_array():
    """ Test custom interpolation function (complex arrays). """
    from CyRK.array import interp_complex_array

    t = np.linspace(0., 1000., 500, dtype=np.float64)
    t_small = np.linspace(0., 1000., 50, dtype=np.float64)
    x = 100 * np.cos(t)**2 - 1.j * np.sin(t)

    numpy_interp = np.interp(t_small, t, x)

    cyrk_interp = np.empty(t_small.size, dtype=np.complex128)
    interp_complex_array(t_small, t, x, cyrk_interp)

    np.allclose(numpy_interp, cyrk_interp)
