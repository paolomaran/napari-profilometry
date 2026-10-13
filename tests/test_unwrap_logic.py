
from napari_profilometry._profilometry_widget import _unwrap_stack, _unwrap_single_image, get_wrapped_phases_widget, reshape_stack_widget, OrderDimsStack, UnwrapMethod

def test_unwrap_stack_scikit(make_napari_viewer,qtbot):
    '''
    This test ensures that the unwrap_single_image widget works via scikit.
    No meaningful tests are made on the image; this test ensures that the
    widget runs without raising errors.
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

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    wrapped_image = viewer.layers[-1].data[:,0,:,:]

    result = _unwrap_stack(stack=wrapped_image,max_threads_no=2,calib_data=None,unwrap_method=UnwrapMethod.SCIKIT)

    assert result.shape == wrapped_image.shape


def test_unwrap_stack_puma(make_napari_viewer,qtbot):
    '''
    This test ensures that the unwrap_single_image widget works via scikit.
    No meaningful tests are made on the image; this test ensures that the
    widget runs without raising errors.
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

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    wrapped_image = viewer.layers[-1].data[:,0,:,:]

    result = _unwrap_stack(stack=wrapped_image,max_threads_no=2,calib_data=None,unwrap_method=UnwrapMethod.PUMA)

    assert result.shape == wrapped_image.shape


def test_unwrap_image_scikit(make_napari_viewer,qtbot):
    '''
    This test ensures that the unwrap_single_image widget works via scikit.
    No meaningful tests are made on the image; this test ensures that the
    widget runs without raising errors.
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

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    wrapped_image = viewer.layers[-1].data[0,0,:,:]

    result = _unwrap_single_image(stack=wrapped_image,calib_data=None,unwrap_method=UnwrapMethod.SCIKIT)

    assert result.shape == wrapped_image.shape


def test_unwrap_image_puma(make_napari_viewer,qtbot):
    '''
    This test ensures that the unwrap_single_image widget works via scikit.
    No meaningful tests are made on the image; this test ensures that the
    widget runs without raising errors.
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

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    wrapped_image = viewer.layers[-1].data[0,0,:,:]

    result = _unwrap_single_image(stack=wrapped_image,calib_data=None,unwrap_method=UnwrapMethod.PUMA)

    assert result.shape == wrapped_image.shape