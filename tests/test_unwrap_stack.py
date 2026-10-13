

from napari_profilometry._profilometry_widget import unwrap_stack_widget, get_wrapped_phases_widget, reshape_stack_widget, OrderDimsStack, UnwrapMethod


def test_unwrap_single_image_scikit(make_napari_viewer,qtbot):
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
    widget_unwrap = unwrap_stack_widget()

    widget_reshape(viewer,image,OrderDimsStack.tpyx,phases=3,time_points=50)
    reshaped_image = viewer.layers[-1]
    widget_wrap_phase(viewer,reshaped_image)

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    wrapped_image = viewer.layers[-1]
    widget_unwrap(viewer,wrapped_image,calib=None,height_conversion=-1.0,unwrapping_method=UnwrapMethod.SCIKIT,apply_height_conversion=True)

    qtbot.waitUntil(lambda: any(layer.name.startswith('UW_WR_PHASE_') for layer in viewer.layers),timeout=1000000)
    result = viewer.layers[-1].data

    assert result.shape == wrapped_image.data.shape


def test_unwrap_single_image_puma(make_napari_viewer,qtbot):
    '''
    This test ensures that the unwrap_single_image widget works via puma.
    No meaningful tests are made on the image; this test ensures that the
    widget runs without raising errors.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    image = viewer.add_image(data,rgb=False)
    widget_reshape = reshape_stack_widget()
    widget_wrap_phase = get_wrapped_phases_widget()
    widget_unwrap = unwrap_stack_widget()

    widget_reshape(viewer,image,OrderDimsStack.tpyx,phases=3,time_points=50)
    reshaped_image = viewer.layers[-1]
    widget_wrap_phase(viewer,reshaped_image)

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    wrapped_image = viewer.layers[-1]
    widget_unwrap(viewer,wrapped_image,calib=None,height_conversion=-1.0,unwrapping_method=UnwrapMethod.PUMA,apply_height_conversion=True)

    qtbot.waitUntil(lambda: any(layer.name.startswith('UW_WR_PHASE_') for layer in viewer.layers),timeout=1000000)
    result = viewer.layers[-1].data

    assert result.shape == wrapped_image.data.shape


def test_force_continuity(make_napari_viewer,qtbot):
    '''
    This test ensures that the unwrap_single_image widget works via puma.
    No meaningful tests are made on the image; this test ensures that the
    widget runs without raising errors.
    '''

    import numpy as np
    from napari.layers import Shapes

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    image = viewer.add_image(data,rgb=False)
    widget_reshape = reshape_stack_widget()
    widget_wrap_phase = get_wrapped_phases_widget()
    widget_unwrap = unwrap_stack_widget()

    widget_reshape(viewer,image,OrderDimsStack.tpyx,phases=3,time_points=50)
    reshaped_image = viewer.layers[-1]
    widget_wrap_phase(viewer,reshaped_image)

    # wait until the worker finishes and adds the wrapped phase image
    qtbot.waitUntil(lambda: any(layer.name.startswith('WR_PHASE_') for layer in viewer.layers),timeout=10000)

    wrapped_image = viewer.layers[-1]
    sh = wrapped_image.data.shape
    shx = sh[-1]
    shy = sh[-2]
    roi_vertices = np.array([[shy/10,shx/10],[9*shy/10,9*shx/10]],dtype=int)
    roi_layer = viewer.add_shapes(roi_vertices,shape_type='rectangle')

    widget_unwrap(viewer,wrapped_image,calib=None,height_conversion=1.0,unwrapping_method=UnwrapMethod.SCIKIT,apply_height_conversion=False,force_continuity=True,roi_continuity=roi_layer)

    qtbot.waitUntil(lambda: any(layer.name.startswith('UW_WR_PHASE_') for layer in viewer.layers),timeout=1000000)
    result = viewer.layers[-1].data

    assert result.shape == wrapped_image.data.shape
