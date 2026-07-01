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
from napari_profilometry._get_h5_data import get_h5_dataset, get_h5_attr, get_datasets_index_by_name, get_group_name
from napari_profilometry.unwrap_processors.PUMA_unwrap import unwrap_phase_puma
from napari_profilometry.unwrap_processors.PUMA_multiprocessing_unwrap import multiPUMA_unwrap_processor
from napari_profilometry.unwrap_processors.SCIKIT_unwrap import unwrap_phase_scikit
from napari_profilometry.unwrap_processors.SCIKIT_multiprocessing_unwrap import multiSCIKIT_unwrap_processor
from napari_profilometry.unwrap_processors.obtain_wrapped_phase_from_fringes import obtain_wrapped_phase
from napari_profilometry.unwrap_processors.stack_force_continuity import force_continuity_stack
 

def find_max(img):
    return divmod(np.argmax(img), img.shape[1])

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

class UnwrapMethod(Enum):
    SCIKIT = 1
    PUMA = 2

class OrderDimsStack(Enum):
    tpyx = 1
    ptyx = 2
    pyxt = 3
    tyxp = 4
    yxpt = 5
    yxtp = 6
    Fyx = 7 


# TODO a reshape widget from yxp to pyx might also be needed
@magic_factory(call_button = 'Reshape to 4D stack')
def reshape_stack_widget(
    viewer: Viewer,
    image: Image,
    old_order: OrderDimsStack = OrderDimsStack.ptyx,
    phases: int = 3,
    time_points: int = 48
):
    if old_order == OrderDimsStack.tpyx:
        print('WARNING: Order is already correct!!')
        return
    else:
        stack = image.data
        if len(stack.shape) != 4  and old_order != OrderDimsStack.Fyx:
            raise ValueError(f'Stack must have 4 dimensions for normal reshaping, instead has {len(stack.shape)}')
        if len(stack.shape) != 3  and old_order == OrderDimsStack.Fyx:
            raise ValueError(f'Stack must have 3 dimensions for frames to time+phase reshaping, instead has {len(stack.shape)}')

        # target order is tpyx
        if old_order == OrderDimsStack.ptyx:
            stack_reshaped = np.moveaxis(stack,[0,1,2,3],[1,0,2,3])
        elif old_order == OrderDimsStack.pyxt:
            stack_reshaped = np.moveaxis(stack,[0,1,2,3],[1,2,3,0])
        elif old_order == OrderDimsStack.tyxp:
            stack_reshaped = np.moveaxis(stack,[0,1,2,3],[0,2,3,1])    
        elif old_order == OrderDimsStack.yxpt:
            stack_reshaped = np.moveaxis(stack,[0,1,2,3],[2,3,1,0])
        elif old_order == OrderDimsStack.yxtp:
            stack_reshaped = np.moveaxis(stack,[0,1,2,3],[2,3,0,1])
        elif old_order == OrderDimsStack.Fyx:
            (sf,sy,sx) = stack.shape
            if sf != phases*time_points:
                raise ValueError(f'Number of frames must be equal to number of phases times number of time points; total frames = {sf}, phases*times = {phases*time_points}')
            stack_reshaped = np.reshape(stack,(time_points,phases,sy,sx))
        else:
            raise RuntimeError(f'Invalid order code selected: {old_order}')

        old_name = image.name        
        image.data = stack_reshaped
        image.name = 'RESHAPED_'+old_name
        viewer.dims.axis_labels = ['time', 'phase', 'y', 'x']
        
        # viewer.add_image(data=stack_reshaped,name = image.name + '_reshaped')
        return stack_reshaped
     


@magic_factory(call_button = 'Get wrapped phase')
def get_wrapped_phases_widget(
    viewer: Viewer,
    image: Image,
    
):
    data = image.data
    if len(data.shape)!=3:
        raise ValueError(f'Image must be of shape (sp,sy,sx), but this image does not have three dimensions (has {len(data.shape)})')
    
    wrapped = obtain_wrapped_phase(image=data)
    viewer.add_image(wrapped,colormap='twilight_shifted')
    return wrapped


@magic_factory(call_button = 'Unwrap single image')
def unwrap_single_image_widget(
    viewer: Viewer,
    image: Image,

    calib: Image = None,
    height_conversion: float = 1.0,
    unwrapping_method: UnwrapMethod = UnwrapMethod.SCIKIT,
    apply_height_conversion: bool = False,
):
    pass


@magic_factory(call_button = 'Unwrap stack')
def unwrap_stack_widget(
    viewer: Viewer,
    image: Image,

    calib: Image = None,
    height_conversion: float = 1.0,
    unwrapping_method: UnwrapMethod = UnwrapMethod.SCIKIT,
    max_threads_no: int = 1,
    roi: Shapes = None,
    apply_height_conversion: bool = False,
    force_continuity: bool = False
):
    pass


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