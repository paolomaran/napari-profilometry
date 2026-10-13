
import numpy as np
import pytest

from napari_profilometry.unwrap_processors.stack_force_continuity import force_continuity_stack


def test_continuity_invalid_dimensions():
    '''
    Tests that continuity correction requires a 3D stack.
    '''

    data = np.zeros((16,16))

    with pytest.raises(ValueError,match="exactly three dimensions"):
        force_continuity_stack(data,(0,10,0,10))


def test_continuity_invalid_type():
    '''
    Tests that continuity correction rejects input
    that is not a numpy array.
    '''

    data = [[[0.0]*16]*16]*3

    with pytest.raises(TypeError,match="numpy.ndarray"):
        force_continuity_stack(data,(0,10,0,10))


def test_continuity_invalid_roi():
    '''
    Tests that continuity correction rejects an ROI
    with invalid boundaries.
    '''

    data = np.zeros((3,16,16))

    with pytest.raises(ValueError,match="xmin must be smaller"):
        force_continuity_stack(data,(0,10,12,5))

    with pytest.raises(ValueError,match="ymin must be smaller"):
        force_continuity_stack(data,(10,5,0,12))


def test_continuity_zero_first_image():
    '''
    Tests that the mean value of the first image is
    set to zero when requested.
    '''

    data = np.ones((3,16,16))*5.0

    result = force_continuity_stack(data,(0,16,0,16),set_zero_first_image=True)

    assert np.isclose(np.mean(result[0]),0.0)
    assert result.shape == data.shape
