

def test_version_imports():
    '''
    Simple test for verifying whether the package import works correctly by checking
    whether the package import results in a readable package version.
    '''
    import napari_profilometry
    assert hasattr(napari_profilometry, "__version__")

def test_reshape_widget_creation():
    '''
    Simple test for verifying proper widget creation by checking whether
    the widget reshape_stack is not None.
    No actual functionality is tested in this test.
    '''
    from napari_profilometry._profilometry_widget import reshape_stack_widget

    widget = reshape_stack_widget()

    assert widget is not None

def test_get_wrapped_phase_widget_creation():
    '''
    Simple test for verifying proper widget creation by checking whether
    the widget get_wrapped_phase is not None.
    No actual functionality is tested in this test.
    '''
    from napari_profilometry._profilometry_widget import get_wrapped_phases_widget

    widget = get_wrapped_phases_widget()

    assert widget is not None

def test_unwrap_single_image_widget_creation():
    '''
    Simple test for verifying proper widget creation by checking whether
    the widget unwrap_single_image is not None.
    No actual functionality is tested in this test.
    '''
    from napari_profilometry._profilometry_widget import unwrap_single_image_widget

    widget = unwrap_single_image_widget()

    assert widget is not None

def test_unwrap_stack_widget_creation():
    '''
    Simple test for verifying proper widget creation by checking whether
    the widget unwrap_stack is not None.
    No actual functionality is tested in this test.
    '''
    from napari_profilometry._profilometry_widget import unwrap_stack_widget

    widget = unwrap_stack_widget()

    assert widget is not None