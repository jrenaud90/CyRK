import numpy as np

import pytest

def test_array_module():
    """ Test that the functions load correctly. """
    from CyRK.array import interpj, interp_complexj, interp, interp_complex, interp_array, interp_complex_array


@pytest.mark.parametrize('t_test', (250.23, 650.0, 0., 1000., 1050., -30., 0.1, 0.5, 0.75, 1.25))
def test_interp(t_test):
    """ Test custom interpolation function (floats). """
    from CyRK.array import interp

    t = np.linspace(0., 1000., 1001, dtype=np.float64)
    x = 100 * np.cos(t)**2

    numpy_interp = np.interp(t_test, t, x)

    cyrk_interp = interp(t_test, t, x)

    assert np.allclose([numpy_interp], [cyrk_interp])

@pytest.mark.parametrize('t_test', (250.23, 650.0, 0., 1000., 1050., -30., 0.1, 0.5, 0.75, 1.25))
def test_interp_complex(t_test):
    """ Test custom interpolation function (complex). """
    from CyRK.array import interp_complex

    t = np.linspace(0., 1000., 1001, dtype=np.float64)
    x = 100 * np.cos(t)**2 - 1.j * np.sin(t)

    numpy_interp = np.interp(t_test, t, x)

    cyrk_interp = interp_complex(t_test, t, x)

    assert np.allclose([numpy_interp], [cyrk_interp])

@pytest.mark.parametrize('t_test', (250.23, 650.0, 0., 1000., 1050., -30., 0.1, 0.5, 0.75, 1.25))
def test_interp_with_provided_j(t_test):
    """ Test custom interpolation function with provided j. """
    from CyRK.array import interpj, interp_complexj, interp, interp_complex

    # Test float function
    t = np.linspace(0., 1000., 1001, dtype=np.float64)
    x = 100 * np.cos(t)**2

    numpy_interp = np.interp(t_test, t, x)

    # Get j
    cyrk_interp_1, provided_j = interpj(t_test, t, x)

    # Use j
    cyrk_interp_2 = interp(t_test, t, x, provided_j=provided_j)

    # Two cyrk versions should be exactly the same.
    assert cyrk_interp_1 == cyrk_interp_2

    assert np.allclose([numpy_interp], [cyrk_interp_1])
    assert np.allclose([numpy_interp], [cyrk_interp_2])

    # Test complex function
    t_cmp = np.linspace(0., 1000., 1001, dtype=np.float64)
    x_cmp = 100 * np.cos(t_cmp)**2 - 1.j * np.sin(t_cmp)
    t_test_cmp = 250.2235

    numpy_interp_cmp = np.interp(t_test_cmp, t_cmp, x_cmp)

    # Get j
    cyrk_interp_1_cmp, provided_j_cmp = interp_complexj(t_test_cmp, t_cmp, x_cmp)
    # Use j
    cyrk_interp_2_cmp = interp_complex(t_test_cmp, t_cmp, x_cmp, provided_j=provided_j_cmp)

    # Two cyrk versions should be exactly the same.
    assert cyrk_interp_1_cmp == cyrk_interp_2_cmp

    assert np.allclose([numpy_interp_cmp], [cyrk_interp_1_cmp])
    assert np.allclose([numpy_interp_cmp], [cyrk_interp_2_cmp])

def test_interp_array():
    """ Test custom interpolation function (float arrays). """
    from CyRK.array import interp_array

    t = np.linspace(0., 1000., 500, dtype=np.float64)
    t_small = np.linspace(0., 1000., 50, dtype=np.float64)
    x = 100 * np.cos(t)**2

    numpy_interp = np.interp(t_small, t, x)

    cyrk_interp = np.empty(t_small.size, dtype=np.float64)
    interp_array(t_small, t, x, cyrk_interp)

    assert np.allclose(numpy_interp, cyrk_interp)

def test_interp_complex_array():
    """ Test custom interpolation function (complex arrays). """
    from CyRK.array import interp_complex_array

    t = np.linspace(0., 1000., 500, dtype=np.float64)
    t_small = np.linspace(0., 1000., 50, dtype=np.float64)
    x = 100 * np.cos(t)**2 - 1.j * np.sin(t)

    numpy_interp = np.interp(t_small, t, x)

    cyrk_interp = np.empty(t_small.size, dtype=np.complex128)
    interp_complex_array(t_small, t, x, cyrk_interp)

    assert np.allclose(numpy_interp, cyrk_interp)
