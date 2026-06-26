'''
Script containing the source code of the various widgets implemented by this plugin.

Author: Paolo Maran (Politecnico di Milano)
'''
import numpy as np
from magicgui import magic_factory, magicgui
from napari import Viewer
from napari.layers import Image, Shapes
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from enum import Enum
import os
from qtpy.QtWidgets import  QWidget
import pathlib
from ._get_h5_data import get_h5_dataset, get_h5_attr, get_datasets_index_by_name, get_group_name
from .unwrap_processors.PUMA_unwrap import unwrap_phase_puma
from .unwrap_processors.PUMA_multiprocessing_unwrap import multiPUMA_unwrap_processor
from .unwrap_processors.SCIKIT_unwrap import unwrap_phase_scikit
from .unwrap_processors.SCIKIT_multiprocessing_unwrap import multiSCIKIT_unwrap_processor
from .unwrap_processors.obtain_wrapped_phase_from_fringes import obtain_wrapped_phase
from .unwrap_processors.stack_force_continuity import force_continuity_stack
 

def find_max(img):
    return divmod(np.argmax(img), img.shape[1])

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


class H5opener(QWidget):

    full_path = os.path.realpath(__file__)
    _folder, _ = os.path.split(full_path) 
    for level in range(3):
        _folder = os.path.join(_folder, os.pardir)
    
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()
        
    def open_h5_dataset(self, path: pathlib.Path = _folder,
                        dataset:int = 0, 
                        ):
        # open file
        directory, filename = os.path.split(path)
        stack,found = get_h5_dataset(path, dataset)
        
        #updates settings
        measurement_names,_ = get_group_name(path, 'measurement')
        measurement_name = measurement_names[0]
        for key in ['magnification','n','NA','pixelsize','wavelength']:
            val = get_h5_attr(path, key, group = measurement_name)
            if len(val)>0 and hasattr(self,key):
                new_value = val[0]
                setattr(getattr(self,key), 'val', new_value)
                print(f'Updated {key} to: {new_value} ')
        fullname = f'dts{dataset}_{filename}'
        self.show_image(stack, im_name=fullname)
                 
    def show_image(self, image_values, im_name, **kwargs):
        '''
        creates a new Image layer with image_values as data
        or updates an existing layer, if 'hold' in kwargs is True 
        '''
        scale = kwargs['scale'] if 'scale' in kwargs else [1.0] * image_values.ndim
        colormap = kwargs.get('colormap', 'gray')    
        if kwargs.get('hold') is True and im_name in self.viewer.layers:
            layer = self.viewer.layers[im_name]
            layer.data = image_values
            layer.scale = scale
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = im_name,
                                            scale = scale,
                                            colormap = colormap)
        self.center_stack(image_values)
        if kwargs.get('autoscale') is True:
            layer.reset_contrast_limits()
        return layer

    def center_stack(self, image_layer):
        '''
        centers a >3D stack in z,y,x 
        '''
        data = image_layer.data
        if data.ndim >2:
            current_step = list(self.viewer.dims.current_step)
            for dim_idx in [-3,-2,-1]:
                current_step[dim_idx] = data.shape[dim_idx]//2
            self.viewer.dims.current_step = current_step    