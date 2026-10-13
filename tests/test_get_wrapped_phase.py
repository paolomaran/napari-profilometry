
from napari_profilometry._profilometry_widget import get_wrapped_phases_widget, reshape_stack_widget, OrderDimsStack
from napari_profilometry.unwrap_processors.obtain_wrapped_phase_from_fringes import obtain_wrapped_phase
import pytest


def test_wrapped_phase_invalid_dimensions():
    '''
    Tests that wrapped phase retrieval rejects images
    that do not have exactly three dimensions.
    '''
    import numpy as np

    data = np.zeros((16,16))

    with pytest.raises(ValueError,match="exactly three dimensions"):
        obtain_wrapped_phase(data)


def test_wrapped_phase_insufficient_phases():
    '''
    Tests that wrapped phase retrieval requires
    at least three phase-shifted images.
    '''
    import numpy as np

    data = np.zeros((2,16,16))

    with pytest.raises(ValueError,match="at least three images"):
        obtain_wrapped_phase(data)



def test_wrapped_phase_constant_signal():
    '''
    Tests phase retrieval using three synthetic phase-shifted
    images with a known phase.
    '''
    import numpy as np

    phase = 0.5
    data = np.array([10+5*np.cos(phase+2*np.pi*i/3) for i in range(3)])
    data = np.broadcast_to(data[:,np.newaxis,np.newaxis],(3,16,16))

    result = obtain_wrapped_phase(data)

    assert result.shape == (16,16)
    np.testing.assert_allclose(result,phase,atol=1e-10)


def test_get_wrapped_phase_single_img(make_napari_viewer,qtbot):
    '''
    Simple test that verifies that wrapped phase can be obtained
    from a sample fringe image.
    Tests include:
    - checking that the shape is correct
    - checking that the resulting image has values between -pi and pi
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_single_image_small.npy")
    image = viewer.add_image(data,rgb=False)
    widget_reshape = reshape_stack_widget()
    widget_wrap_phase = get_wrapped_phases_widget()

    widget_reshape(viewer,image,OrderDimsStack.Fyx,phases=3,time_points=1)
    reshaped_image = viewer.layers[-1]
    widget_wrap_phase(viewer,reshaped_image)

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    result = viewer.layers[-1].data

    assert np.min(result) >= -np.pi and np.max(result) <= np.pi
    assert result.shape == (1,1,reshaped_image.data.shape[-2],reshaped_image.data.shape[-1])

    
def test_get_wrapped_phase_stack(make_napari_viewer,qtbot):
    '''
    Simple test that verifies that wrapped phase can be obtained 
    from a sample fringe image stack.
    Tests include:
    - checking that the shape is correct
    - checking that the resulting image has values between -pi and pi
    '''

    import numpy as np
    
    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    image = viewer.add_image(data,rgb=False)
    widget_reshape = reshape_stack_widget()
    widget_wrap_phase = get_wrapped_phases_widget()

    widget_reshape(viewer,image,OrderDimsStack.tpyx,phases=3,time_points=50)
    reshaped_image = viewer.layers[-1]
    widget_wrap_phase(viewer,reshaped_image)

    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    result = viewer.layers[-1].data

    assert np.min(result) >= -np.pi and np.max(result) <= +np.pi
    assert result.shape == (reshaped_image.data.shape[0],1,reshaped_image.data.shape[-2],reshaped_image.data.shape[-1])