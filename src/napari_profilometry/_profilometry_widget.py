'''
Script containing the source code of the various widgets implemented by this plugin.

Author: Paolo Maran, Andrea Bassi (Politecnico di Milano)
'''
import numpy as np
from magicgui import magic_factory, magicgui
from napari import Viewer
from napari.layers import Image, Shapes
from napari.qt.threading import thread_worker
# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
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
 

# def find_max(img):
#     return divmod(np.argmax(img), img.shape[1])

# def find_nearest(array, value):
#     idx = (np.abs(array - value)).argmin()
#     return idx

_active_thread_workers = {
    'get_wrapped_phase': None,
    'unwrap_single_image': None,
    'unwrap_stack': None
}

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
    yxp = 8


@magic_factory(call_button = 'Reshape to 4D stack')
def reshape_stack_widget(
    viewer: Viewer,
    image: Image,
    old_order: OrderDimsStack = OrderDimsStack.tpyx,
    phases: int = 3,
    time_points: int = 50
):
    valid_orders_4d = [ OrderDimsStack.ptyx,
                        OrderDimsStack.tpyx,
                        OrderDimsStack.pyxt,
                        OrderDimsStack.tyxp,
                        OrderDimsStack.yxpt,
                        OrderDimsStack.yxtp]
    
    valid_orders_3d = [ OrderDimsStack.Fyx,
                        OrderDimsStack.yxp]
    
    stack = image.data

    if old_order in valid_orders_4d:
        if len(stack.shape) != 4:
            raise ValueError(f'Stack must have 4 dimensions for selected reshaping, instead has {len(stack.shape)}')
    elif old_order in valid_orders_3d:
        if len(stack.shape) != 3:
            raise ValueError(f'Stack must have 3 dimensions for selected reshaping, instead has {len(stack.shape)}')
    else:
        raise ValueError('Unknown or unrecognized reshape order')
    
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
    elif old_order == OrderDimsStack.yxp:
        stack_reshaped = np.moveaxis(stack,[0,1,2],[2,0,1])
    elif old_order == OrderDimsStack.tpyx:
        print('WARNING: Order is already correct!!')
        stack_reshaped = stack
    else:
        raise RuntimeError(f'Invalid order code selected: {old_order}')

    old_name = image.name        
    image.data = stack_reshaped
    image.name = 'RESHAPED_'+old_name
    viewer.dims.axis_labels = ['time', 'phase', 'y', 'x']
    
    # viewer.add_image(data=stack_reshaped,name = image.name + '_reshaped')
    return stack_reshaped
     

@thread_worker
def _tw_get_wrapped_phases(data):
    '''
    Implements get_wrapped_phase_widget as a thread worker.
    '''

    if len(data.shape) == 3:
        wrapped = obtain_wrapped_phase(image=data)
        wrapped = wrapped[np.newaxis,np.newaxis,:,:]
    
    if len(data.shape) == 4:
        (t,_,_,_) = data.shape
        wrapped = np.empty_like(data[:,0,:,:],dtype = np.float64)
        print(wrapped.shape)
        for time in range(t):
            wrapped[time] = obtain_wrapped_phase(data[time])
        wrapped = wrapped[:,np.newaxis,:,:]
    
    return wrapped


def _get_wrapped_phases_init(widget):
    # magic_factory calls this once, right after building the FunctionGui
    # instance, and hands us that instance -- this is how we get a real
    # handle on it (with .call_button etc.), instead of the bare factory.
    _active_thread_workers['get_wrapped_phase'] = widget

@magic_factory(call_button = 'Get wrapped phase', widget_init=_get_wrapped_phases_init)
def get_wrapped_phases_widget(
    viewer: Viewer,
    image: Image,
):
    this_widget = _active_thread_workers['get_wrapped_phase']  # the real FunctionGui
    
    def _on_done(wr_phase):
        viewer.add_image(wr_phase,name = 'WR_PHASE_'+image.name,colormap='twilight_shifted')
        this_widget.call_button.enabled = True
        return wr_phase
    
    data = image.data
    if len(data.shape) != 3 and len(data.shape) != 4:
        raise ValueError(f'Image must be of shape (sp,sy,sx) or (t,sp,sy,sx), but this image is not (has {len(data.shape)} dimensions)')
    
    this_widget.call_button.enabled = False
    worker = _tw_get_wrapped_phases(data)
    worker.returned.connect(_on_done)
    worker.start()    
            


@thread_worker
def _tw_unwrap_single_image(stack,calib_data,unwrap_method):
    if unwrap_method == UnwrapMethod.SCIKIT:
        uw_stack = unwrap_phase_scikit(
            stack,
            calib=calib_data)
    elif unwrap_method == UnwrapMethod.PUMA:
        uw_stack = unwrap_phase_puma(
            stack,
            calib=calib_data)
        
    return uw_stack



def _unwrap_single_image_init(widget):
    _active_thread_workers['unwrap_single_image'] = widget


@magic_factory(call_button = 'Unwrap single image',widget_init=_unwrap_single_image_init)
def unwrap_single_image_widget(
    viewer: Viewer,
    image: Image,
    calib: Image = None,
    height_conversion: float = 1.0,
    unwrapping_method: UnwrapMethod = UnwrapMethod.SCIKIT,
    apply_height_conversion: bool = False,
):
    
    this_widget = _active_thread_workers['unwrap_single_image']
    
    def _on_done(uw_stack):
        if apply_height_conversion:
            uw_stack = uw_stack * height_conversion
        uw_stack = uw_stack[np.newaxis,np.newaxis,:,:]
        viewer.add_image(uw_stack,name = 'UW_'+image.name,colormap = 'gray')
        this_widget.call_button.enabled = True
        return uw_stack  
    
    stack = image.data.squeeze()
    calib_data = calib.data if calib is not None else None

    if len(stack.shape) > 2:
        raise RuntimeError('Image must be bidimensional, phase-wrapped data.')
    
    this_widget.call_button.enabled = False
    worker = _tw_unwrap_single_image(stack,calib_data,unwrapping_method)
    worker.returned.connect(_on_done)
    worker.start()



@thread_worker
def _tw_unwrap_stack(stack,max_threads_no,calib_data,unwrap_method):
    if unwrap_method == UnwrapMethod.SCIKIT:
        uw_stack = multiSCIKIT_unwrap_processor(
            stack,
            max_threads=max_threads_no, 
            calib=calib_data)
    elif unwrap_method == UnwrapMethod.PUMA:
        uw_stack = multiPUMA_unwrap_processor(
            stack,
            max_threads=max_threads_no, 
            calib=calib_data)
    return uw_stack


def _tw_continuity_forcer(uw_stack,roi,set_zero):
    pass

def _unwrap_stack_init(widget):
    _active_thread_workers['unwrap_stack'] = widget


@magic_factory(call_button = 'Unwrap stack',widget_init=_unwrap_stack_init)
def unwrap_stack_widget(
    viewer: Viewer,
    image: Image,
    calib: Image = None,
    height_conversion: float = 0.442,
    unwrapping_method: UnwrapMethod = UnwrapMethod.SCIKIT,
    max_threads_no: int = 1,
    apply_height_conversion: bool = False,
    force_continuity: bool = False,
    roi_continuity: Shapes = None,
):
    
    this_widget = _active_thread_workers['unwrap_stack']

    def _on_done(uw_stack):
        if force_continuity:
            if roi_continuity is not None:
                ymin = int(np.min(roi_continuity.data[0][...,-2]))
                ymax = int(np.max(roi_continuity.data[0][...,-2]))
                xmin = int(np.min(roi_continuity.data[0][...,-1]))
                xmax = int(np.max(roi_continuity.data[0][...,-1]))
            else:
                ymin = 0
                ymax = sy-1
                xmin = 0
                xmax = sx-1
            uw_stack = force_continuity_stack(
                uw_stack,
                (ymin,ymax,xmin,xmax),
                set_zero_first_image = True)
    
        if apply_height_conversion:
            uw_stack = uw_stack * height_conversion

        viewer.add_image(uw_stack[:,np.newaxis,:,:],name = 'UW_'+image.name,colormap = 'gray')
        this_widget.call_button.enabled = True
        return uw_stack

    stack = image.data.squeeze()
    calib_data = calib.data.squeeze() if calib is not None else None

    (st,sy,sx) = stack.shape

    this_widget.call_button.enabled = False
    worker_uw = _tw_unwrap_stack(stack,max_threads_no,calib_data,unwrapping_method)
    worker_uw.returned.connect(_on_done)
    worker_uw.start()
    


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