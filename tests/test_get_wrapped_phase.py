
from napari_profilometry._profilometry_widget import get_wrapped_phases_widget, reshape_stack_widget, OrderDimsStack


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