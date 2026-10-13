
import numpy as np
import pytest

from napari_profilometry.unwrap_processors.PUMA_unwrap import unwrap_phase_puma, mincut,clique_energy_ho


def test_puma_invalid_type():
    '''
    Tests that PUMA rejects input that is not a numpy.ndarray.
    '''

    data = [[0.0]*16]*16

    with pytest.raises(TypeError,match="numpy.ndarray"):
        unwrap_phase_puma(data)


def test_puma_invalid_dimensions():
    '''
    Tests that PUMA requires exactly two dimensions.
    '''

    data = np.zeros((3,16,16))

    with pytest.raises(ValueError,match="exactly two dimensions"):
        unwrap_phase_puma(data)


def test_puma_invalid_dtype():
    '''
    Tests that PUMA rejects images containing non-numeric values.
    '''

    data = np.full((16,16),'invalid')

    with pytest.raises(TypeError,match="convertible"):
        unwrap_phase_puma(data)


def test_puma_invalid_calibration_type():
    '''
    Tests that PUMA requires a numpy.ndarray calibration image.
    '''

    data = np.zeros((16,16))
    calib = [[0.0]*16]*16

    with pytest.raises(TypeError,match="Calibration image must be"):
        unwrap_phase_puma(data,calib=calib)


def test_puma_invalid_calibration_shape():
    '''
    Tests that the calibration image must have the same
    dimensions as the wrapped phase.
    '''

    data = np.zeros((16,16))
    calib = np.zeros((8,8))

    with pytest.raises(ValueError,match="Calib must have same shape"):
        unwrap_phase_puma(data,calib=calib)


def test_puma_invalid_calibration_dtype():
    '''
    Tests that PUMA rejects non-numeric calibration images.
    '''

    data = np.zeros((16,16))
    calib = np.full((16,16),'invalid')

    with pytest.raises(TypeError,match="Calibration image elements"):
        unwrap_phase_puma(data,calib=calib)


def test_puma_clique_energy():
    '''
    Tests PUMA clique energy calculation for values
    below and above the quadratic threshold.
    '''

    data = np.array([0.5,1.0,2.0,4.0])

    result = clique_energy_ho(data,p=1.0,th=2.0,quant='no')
    expected = np.array([0.125,0.5,2.0,4.0])

    np.testing.assert_allclose(result,expected)


def test_puma_clique_energy_invalid_quant():
    '''
    Tests that an unrecognized quantization mode
    raises a ValueError.
    '''

    data = np.array([1.0,2.0])

    with pytest.raises(ValueError,match="quant must be"):
        clique_energy_ho(data,p=1.0,th=2.0,quant='invalid')


def test_puma_mincut():
    '''
    Tests that PUMA graph-cut calculation produces
    the correct minimum cut for two disconnected nodes.
    '''

    sourcesink = np.array([
        [0,10,0],
        [1,0,10]
    ])

    flow,cutside = mincut(sourcesink,[])

    assert np.isclose(flow,0.0)
    assert cutside.shape == (2,2)
    assert cutside[0,1] != cutside[1,1]