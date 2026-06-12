from src.napari_profilometry._reshape_profilometry_stacks import reshape_stack
from src.napari_profilometry._extract_phase_widget import extract_phase
from src.napari_profilometry._unwrap_widget import unwrap_image
from src.napari_profilometry._stack_unwrap_widget import unwrap_stack

'''
Script that runs the napari widget from the IDE. 
It is not executed when the plugin runs.
'''

if __name__ == '__main__':
    import napari
    viewer = napari.Viewer()
    widget_uw_stack = unwrap_stack(viewer)
    widget_uw = unwrap_image()
    widget_extract_phase = extract_phase()
    widget_reshape = reshape_stack()    

    w1 = viewer.window.add_dock_widget(widget_reshape, 
                                       name = 'Reshape stack', 
                                       add_vertical_stretch = True)
    w2 = viewer.window.add_dock_widget(widget_extract_phase, 
                                       name = 'Extract phase', 
                                       add_vertical_stretch = True)
    w3 = viewer.window.add_dock_widget(widget_uw, 
                                       name = 'Unwrap image', 
                                       add_vertical_stretch = True)
    w4 = viewer.window.add_dock_widget(widget_uw_stack, 
                                       name = 'Unwrap image stack', 
                                       add_vertical_stretch = True)
    viewer.window._qt_window.tabifyDockWidget(w1, w2, w3, w4)
    napari.run()
