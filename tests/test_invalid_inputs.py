
import numpy as np
import pytest

from napari_profilometry._profilometry_widget import (
    reshape_stack_widget,
    get_wrapped_phases_widget,
    unwrap_single_image_widget,
    OrderDimsStack,
)


def test_reshape_invalid_4d_input(make_napari_viewer):
    '''
    Tests that selecting a 4D reshape order with 3D input
    raises a ValueError.
    '''

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((150,16,16)),rgb=False)
    widget = reshape_stack_widget()

    with pytest.raises(ValueError,match="Stack must have 4 dimensions"):
        widget(viewer,image,old_order=OrderDimsStack.ptyx)


def test_reshape_invalid_3d_input(make_napari_viewer):
    '''
    Tests that selecting a 3D reshape order with 4D input
    raises a ValueError.
    '''

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((50,3,16,16)),rgb=False)
    widget = reshape_stack_widget()

    with pytest.raises(ValueError,match="Stack must have 3 dimensions"):
        widget(viewer,image,old_order=OrderDimsStack.Fyx)


def test_reshape_invalid_frame_count(make_napari_viewer):
    '''
    Tests that the number of frames must match the number
    of phases multiplied by the number of time points.
    '''

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((100,16,16)),rgb=False)
    widget = reshape_stack_widget()

    with pytest.raises(ValueError,match="Number of frames must be equal"):
        widget(viewer,image,old_order=OrderDimsStack.Fyx,phases=3,time_points=50)


def test_get_wrapped_phases_invalid_dimensions(make_napari_viewer):
    '''
    Tests that wrapped phase retrieval rejects images
    that are neither 3D nor 4D.
    '''

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((16,16)),rgb=False)
    widget = get_wrapped_phases_widget()

    with pytest.raises(ValueError,match="Image must be of shape"):
        widget(viewer,image)


def test_unwrap_single_image_invalid_dimensions(make_napari_viewer):
    '''
    Tests that single image unwrapping rejects input
    containing more than two dimensions.
    '''

    viewer = make_napari_viewer()
    image = viewer.add_image(np.zeros((3,16,16)),rgb=False)
    widget = unwrap_single_image_widget()

    with pytest.raises(RuntimeError,match="Image must be bidimensional"):
        widget(viewer,image)
