'''
Script for Phase Unwrapping via Max-flow/Min-cut (PUMA) a stack of images.
Reference: Bioucas-Dias and Valadão, IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 16, NO. 3, MARCH 2007.

Due to the large computational cost of unwrapping images with PUMA,
a multiprocessing approach is used. In this way, image processing is 
parallelized without needing GPU usage.

Author: Anik Ghosh, Politecnico di Milano
'''
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support, cpu_count
from itertools import repeat
from .PUMA_unwrap import funcPUMA as unwrap_single_image



def multiPUMA_unwrap(stack: np.ndarray, max_threads: int = None, calib: np.ndarray = None):
    '''
    Function that, given an image stack of wrapped phase maps,
    unwraps each image using the PUMA algorithm.

    Parameters
    ----------
    stack: ndarray (sz x sy x sx)
        The image array containing the wrapped phase.
        Must be ordered as (z, y, x).
        Element type must be float.
    max_threads: int
        Number of maximum threads used in multiprocessing.
        Must be an integer greater than zero and smaller than the maximum
        number of threads available on the current machine. 
        If undefined, defaults to half of the maximum available threads.
    calib: ndarray (sy,sx)
        Plane calibration. This will be subtracted from all images in stack.
        Therefore, it must have the same shape (in y and x) as stack.
    
    Returns
    -------
    uw_stack: ndarray (sz x sy x sx)
        Image stack of same dimensions as stack, containing the PUMA reconstructions
    '''

    # Input check: check that stack is of correct type and shape
    if type(stack) != np.ndarray:
        raise TypeError(f'Stack must be a numpy.ndarray, is instead {type(stack)}')
    l = len(np.shape(stack))
    if  l != 3:
        raise ValueError(f'Stack must have exactly three dimensions, has instead {l}')
    if stack.dtype != np.float64:
        try:
            stack = np.float64(stack)
        except:
            raise TypeError(f'Stack elements must be numpy.float64 or convertible to it, are instead {stack.dtype}')

    # Input check: max_threads must be an int between 1 and maximum available threads
    max_processes = cpu_count()
    if max_threads == None:
        max_threads = int(max_processes/2)
    else:
        if type(max_threads) != int:
            try:
                max_threads = int(max_threads)
            except:
                raise TypeError(f'max_threads must be of type int, is instead {type(max_threads)}')
            
        if max_threads <= 0 or max_threads >= max_processes:
            raise ValueError(f'max_threads must be between 1 and {max_processes-1} (inclusive), is instead {max_threads}')
    
    # Input check: calib must be a ndarray of float64 with same (sy,sx) as stack
    if calib == None:
        calib = np.zeros_like(stack[0])
    else:
        if type(calib) != np.ndarray:
            raise TypeError(f'Calibration image must be a numpy.ndarray, is instead {type(calib)}')
        if calib.dtype != np.float64:
            try:
                calib = np.float64(calib)
            except:
                raise TypeError(f'Calibration image elements must be numpy.float64 or convertible to it, are instead {calib.dtype}')
        if np.shape(calib) != np.shape(stack[0]):
            raise ValueError(f'Calib must have same shape as stack slices; instead, calib is {np.shape(calib)} while stack slices are {np.shape(stack[0])}')
            
