

from napari_profilometry._profilometry_widget import reshape_stack_widget,OrderDimsStack

def test_reshape_stack_tpyx(make_napari_viewer):
    '''
    Tests widget's tpyx->tpyx reshaping on a sample dataset (no operation).
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    exp_results = data.copy()
    image = viewer.add_image(data,rgb=False)
    widget = reshape_stack_widget()

    results = widget(
        viewer,image,old_order = OrderDimsStack.tpyx,
        phases=3,time_points=50)

    assert viewer.dims.axis_labels == ('time', 'phase', 'y', 'x')
    assert results.shape == (50,3,66,114)
    np.testing.assert_array_equal(results, exp_results)


def test_reshape_stack_ptyx(make_napari_viewer):
    '''
    Tests widget's ptyx->tpyx reshaping on a sample dataset.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    exp_results = data.copy()
    # data is nominally in tpxy, so we need to put it out of order first
    data = np.moveaxis(data,[0,1,2,3],[1,0,2,3])
    image = viewer.add_image(data,rgb=False)
    widget = reshape_stack_widget()

    results = widget(
        viewer,image,old_order = OrderDimsStack.ptyx,
        phases=3,time_points=50)

    assert viewer.dims.axis_labels == ('time', 'phase', 'y', 'x')
    assert results.shape == (50,3,66,114)
    np.testing.assert_array_equal(results, exp_results)


def test_reshape_stack_pyxt(make_napari_viewer):
    '''
    Tests widget's pyxt->tpyx reshaping on a sample dataset.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    exp_results = data.copy()
    # data is nominally in tpyx, so we need to put it out of order first
    data = np.moveaxis(data,[0,1,2,3],[3,0,1,2])
    image = viewer.add_image(data,rgb=False)
    widget = reshape_stack_widget()

    results = widget(
        viewer,image,old_order = OrderDimsStack.pyxt,
        phases=3,time_points=50)

    assert viewer.dims.axis_labels == ('time', 'phase', 'y', 'x')
    assert results.shape == (50,3,66,114)
    np.testing.assert_array_equal(results, exp_results)


def test_reshape_stack_tyxp(make_napari_viewer):
    '''
    Tests widget's tyxp->tpyx reshaping on a sample dataset.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    exp_results = data.copy()
    # data is nominally in tpyx, so we need to put it out of order first
    data = np.moveaxis(data,[0,1,2,3],[0,3,1,2])
    image = viewer.add_image(data,rgb=False)
    widget = reshape_stack_widget()

    results = widget(
        viewer,image,old_order = OrderDimsStack.tyxp,
        phases=3,time_points=50)

    assert viewer.dims.axis_labels == ('time', 'phase', 'y', 'x')
    assert results.shape == (50,3,66,114)
    np.testing.assert_array_equal(results, exp_results)


def test_reshape_stack_yxpt(make_napari_viewer):
    '''
    Tests widget's yxpt->tpyx reshaping on a sample dataset.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    exp_results = data.copy()
    # data is nominally in tpyx, so we need to put it out of order first
    data = np.moveaxis(data,[0,1,2,3],[3,2,0,1])
    image = viewer.add_image(data,rgb=False)
    widget = reshape_stack_widget()

    results = widget(
        viewer,image,old_order = OrderDimsStack.yxpt,
        phases=3,time_points=50)

    assert viewer.dims.axis_labels == ('time', 'phase', 'y', 'x')
    assert results.shape == (50,3,66,114)
    np.testing.assert_array_equal(results, exp_results)


def test_reshape_stack_yxtp(make_napari_viewer):
    '''
    Tests widget's yxtp->tpyx reshaping on a sample dataset.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    exp_results = data.copy()
    # data is nominally in tpyx, so we need to put it out of order first
    data = np.moveaxis(data,[0,1,2,3],[2,3,0,1])
    image = viewer.add_image(data,rgb=False)
    widget = reshape_stack_widget()

    results = widget(
        viewer,image,old_order = OrderDimsStack.yxtp,
        phases=3,time_points=50)

    assert viewer.dims.axis_labels == ('time', 'phase', 'y', 'x')
    assert results.shape == (50,3,66,114)
    np.testing.assert_array_equal(results, exp_results)


def test_reshape_stack_Fyx(make_napari_viewer):
    '''
    Tests widget's Fyx->tpyx reshaping on a sample dataset.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    data = np.load("samples/ex_stack.npy")
    exp_results = data.copy()
    # data is nominally in tpyx, so we need to flatten time and phase first
    data = np.reshape(data,(150,66,114))
    image = viewer.add_image(data,rgb=False)
    widget = reshape_stack_widget()

    results = widget(
        viewer,image,old_order = OrderDimsStack.Fyx,
        phases=3,time_points=50)

    assert viewer.dims.axis_labels == ('time', 'phase', 'y', 'x')
    assert results.shape == (50,3,66,114)
    np.testing.assert_array_equal(results, exp_results)
