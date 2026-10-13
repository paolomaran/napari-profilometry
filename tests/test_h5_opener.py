
from napari_profilometry._profilometry_widget import H5opener


def test_h5_opener_creation(make_napari_viewer):
    '''
    Tests that the H5 opener widget can be created.
    '''

    viewer = make_napari_viewer()
    widget = H5opener(viewer)

    assert widget is not None
    assert widget.viewer is viewer


def test_h5_opener_load_dataset(make_napari_viewer,sample_h5):
    '''
    Tests loading an HDF5 dataset into napari.
    Tests include:
    - checking that an image layer is created
    - checking that the dataset name is correct
    - checking the shape and values of the imported data
    '''

    import numpy as np

    viewer = make_napari_viewer()
    widget = H5opener(viewer)

    widget.open_h5_dataset(sample_h5,dataset=0)

    assert len(viewer.layers) == 1

    image = viewer.layers[0]

    assert image.name == 'dts0_test_measurement.h5'
    assert image.data.shape == (3,6,8)
    assert image.rgb is False

    expected = np.arange(3*6*8,dtype=np.float32).reshape(3,6,8)
    np.testing.assert_array_equal(image.data,expected)


def test_h5_opener_metadata(make_napari_viewer,sample_h5):
    '''
    Tests that measurement attributes are correctly imported
    into the H5 opener widget.
    '''

    from types import SimpleNamespace

    viewer = make_napari_viewer()
    widget = H5opener(viewer)

    widget.magnification = SimpleNamespace(val=None)
    widget.pixelsize = SimpleNamespace(val=None)
    widget.wavelength = SimpleNamespace(val=None)

    widget.open_h5_dataset(sample_h5,dataset=0)

    assert widget.magnification.val == 20.0
    assert widget.pixelsize.val == 0.5
    assert widget.wavelength.val == 0.532


def test_h5_opener_show_image(make_napari_viewer):
    '''
    Tests adding an image to napari using the H5 opener.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    widget = H5opener(viewer)

    data = np.ones((3,6,8))
    image = widget.show_image(data,im_name='test_image')

    assert len(viewer.layers) == 1
    assert image.name == 'test_image'
    np.testing.assert_array_equal(image.data,data)


def test_h5_opener_hold_image(make_napari_viewer):
    '''
    Tests updating an existing layer instead of creating a new one.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    widget = H5opener(viewer)

    data0 = np.ones((3,6,8))
    data1 = np.ones((3,6,8))*2

    image0 = widget.show_image(data0,im_name='test_image')
    image1 = widget.show_image(data1,im_name='test_image',hold=True)

    assert len(viewer.layers) == 1
    assert image0 is image1
    np.testing.assert_array_equal(image1.data,data1)


def test_h5_opener_center_stack(make_napari_viewer):
    '''
    Tests that the napari viewer is centered on the loaded stack.
    '''

    import numpy as np

    viewer = make_napari_viewer()
    widget = H5opener(viewer)

    data = np.ones((5,10,20))
    widget.show_image(data,im_name='test_image')

    assert tuple(viewer.dims.current_step) == (2,5,10)
