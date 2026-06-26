'''
Simple processor for unwrapping images via SCIKIT.

Author: Paolo Maran, Politecnico di Milano
'''

import skimage.restoration as scikit
import numpy as np

def unwrap_phase_scikit(wrapped_phase,calib=None):
    '''
    Processor that, given an image of a wrapped phase maps,
    unwraps each image using the SCIKIT implemented algorithm.

    Parameters
    ----------
    wrapped_phase: ndarray (sy x sx)
        The image array containing the wrapped phase.
        Element type must be float.
    calib: ndarray (sy,sx), Optional
        Plane calibration. This will be subtracted from all images in stack.
        Therefore, it must have the same shape (in y and x) as stack.
    
    Returns
    -------
    uw_image: ndarray (sy x sx)
        Image of same dimensions as wrapped_phase, containing the SCIKIT reconstruction
    '''

    
    # Input check: check that wrapped_phase is of correct type and shape
    if type(wrapped_phase) is not np.ndarray:
        raise TypeError(f'Stack must be a numpy.ndarray, is instead {type(wrapped_phase)}')
    ndims = len(np.shape(wrapped_phase))
    if  ndims != 2:
        raise ValueError(f'Stack must have exactly two dimensions, has instead {ndims}')
    if wrapped_phase.dtype != np.float64:
        try:
            wrapped_phase = np.float64(wrapped_phase)
        except Exception:
            raise TypeError(f'Stack elements must be numpy.float64 or convertible to it, are instead {wrapped_phase.dtype}')
        
    
    # Input check: calib must be a ndarray of float64 with same (sy,sx) as stack
    if calib is None:
        calib = np.zeros_like(wrapped_phase)
    else:
        if type(calib) is not np.ndarray:
            raise TypeError(f'Calibration image must be a numpy.ndarray, is instead {type(calib)}')
        if calib.dtype != np.float64:
            try:
                calib = np.float64(calib)
            except Exception:
                raise TypeError(f'Calibration image elements must be numpy.float64 or convertible to it, are instead {calib.dtype}')
        if np.shape(calib) != np.shape(wrapped_phase):
            raise ValueError(f'Calib must have same shape as wrapped_phase; instead, calib is {np.shape(calib)} while stack slices are {np.shape(wrapped_phase)}')
        


    uw_img = scikit.unwrap_phase(wrapped_phase)
    if calib is not None:
        uw_img = uw_img - calib

    return uw_img