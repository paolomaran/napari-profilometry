from src.napari_profilometry._profilometry_widget import reshape_stack_widget, \
    get_wrapped_phases_widget, unwrap_single_image_widget, unwrap_stack_widget

'''
Script that runs the napari widget from the IDE. 
It is not executed when the plugin runs.
'''

if __name__ == '__main__':
    import napari
    from magicgui import magicgui
    viewer = napari.Viewer()

    widget_reshape = reshape_stack_widget() 
    widget_extract_phase = get_wrapped_phases_widget()
    widget_uw = unwrap_single_image_widget()
    widget_uw_stack = unwrap_stack_widget()
    

    w1 = viewer.window.add_dock_widget(widget_reshape, 
                                       name = 'Reshape stack', 
                                       add_vertical_stretch = True,
                                       tabify = True)
    w2 = viewer.window.add_dock_widget(widget_extract_phase, 
                                       name = 'Extract phase', 
                                       add_vertical_stretch = True,
                                       tabify=True)
    w3 = viewer.window.add_dock_widget(widget_uw, 
                                       name = 'Unwrap single frame', 
                                       add_vertical_stretch = True)
    w4 = viewer.window.add_dock_widget(widget_uw_stack, 
                                       name = 'Unwrap image stack', 
                                       add_vertical_stretch = True,
                                       tabify = True)
    napari.run()
