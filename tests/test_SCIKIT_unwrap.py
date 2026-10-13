
import numpy as np
import pytest

from napari_profilometry.unwrap_processors.SCIKIT_unwrap import unwrap_phase_scikit


def test_scikit_unwrap_known_phase():
    '''
    Tests SCIKIT unwrapping using a synthetic continuous phase map.
    Tests include:
    - checking that the shape is correct
    - checking that the unwrapped phase matches the original modulo 2pi
    '''

    y,x = np.mgrid[0:32,0:32]
    expected = 0.4*x + 0.2*y
    wrapped = np.angle(np.exp(1j*expected))

    result = unwrap_phase_scikit(wrapped)

    assert result.shape == expected.shape
    np.testing.assert_allclose(np.diff(result,axis=0),np.diff(expected,axis=0),atol=1e-10)
    np.testing.assert_allclose(np.diff(result,axis=1),np.diff(expected,axis=1),atol=1e-10)


def test_scikit_unwrap_calibration():
    '''
    Tests that SCIKIT unwrapping correctly subtracts
    the supplied calibration image.
    '''

    y,x = np.mgrid[0:32,0:32]
    phase = 0.4*x + 0.2*y
    wrapped = np.angle(np.exp(1j*phase))
    calib = np.ones((32,32))*0.5

    reference = unwrap_phase_scikit(wrapped)
    result = unwrap_phase_scikit(wrapped,calib=calib)

    np.testing.assert_allclose(result,reference-calib,atol=1e-10)


def test_scikit_unwrap_integer_input():
    '''
    Tests that integer input is converted to float64.
    '''

    data = np.zeros((16,16),dtype=np.uint16)

    result = unwrap_phase_scikit(data)

    assert result.dtype == np.float64
    assert result.shape == (16,16)


def test_scikit_unwrap_invalid_type():
    '''
    Tests that input that is not a numpy.ndarray
    raises a TypeError.
    '''

    data = [[0.0]*16]*16

    with pytest.raises(TypeError,match="numpy.ndarray"):
        unwrap_phase_scikit(data)


def test_scikit_unwrap_invalid_dimensions():
    '''
    Tests that SCIKIT unwrapping requires a 2D image.
    '''

    data = np.zeros((3,16,16))

    with pytest.raises(ValueError,match="exactly two dimensions"):
        unwrap_phase_scikit(data)


def test_scikit_unwrap_invalid_dtype():
    '''
    Tests that input containing non-numeric values
    raises a TypeError.
    '''

    data = np.full((16,16),'invalid')

    with pytest.raises(TypeError,match="convertible"):
        unwrap_phase_scikit(data)


def test_scikit_unwrap_invalid_calibration_type():
    '''
    Tests that calibration data must be a numpy.ndarray.
    '''

    data = np.zeros((16,16))

    with pytest.raises(TypeError,match="Calibration image must be"):
        unwrap_phase_scikit(data,calib=[[0.0]*16]*16)


def test_scikit_unwrap_invalid_calibration_shape():
    '''
    Tests that the calibration image has the same
    dimensions as the wrapped phase.
    '''

    data = np.zeros((16,16))
    calib = np.zeros((8,8))

    with pytest.raises(ValueError,match="Calib must have same shape"):
        unwrap_phase_scikit(data,calib=calib)


def test_scikit_unwrap_calibration_conversion():
    '''
    Tests that integer calibration data is converted
    into float64 before subtraction.
    '''

    data = np.zeros((16,16))
    calib = np.ones((16,16),dtype=np.uint16)

    result = unwrap_phase_scikit(data,calib=calib)

    np.testing.assert_allclose(result,-1.0)


def test_scikit_unwrap_invalid_calibration_dtype():
    '''
    Tests that non-numeric calibration values
    raise a TypeError.
    '''

    data = np.zeros((16,16))
    calib = np.full((16,16),'invalid')

    with pytest.raises(TypeError,match="Calibration image elements"):
        unwrap_phase_scikit(data,calib=calib)
