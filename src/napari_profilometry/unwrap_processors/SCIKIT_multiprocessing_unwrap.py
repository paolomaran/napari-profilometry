import skimage.restoration as scikit
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support, cpu_count
from itertools import repeat


def _unwrap_phase_scikit_single(wrapped_phase,calib_phase: None,img_index: int=-1, img_total: int=-1):
    uw_img = scikit.unwrap_phase(wrapped_phase)
    if calib_phase is not None:
        uw_img = uw_img - calib_phase

    if img_index != -1 and img_total != -1:
        print(f'[{(100*(img_index+1)/img_total):.2f}% processed]')

    return uw_img


def multiSCIKIT_unwrap_processor(stack: np.ndarray, max_threads: int = None, calib: np.ndarray = None):  # noqa: N802
    '''
    Processor that, given an image stack of wrapped phase maps,
    unwraps each image using the PUMA algorithm.
    Due to very high computational demand and long execution time, this
    function MUST be called always in a separate thread.

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
    if type(stack) is not np.ndarray:
        raise TypeError(f'Stack must be a numpy.ndarray, is instead {type(stack)}')
    ndims = len(np.shape(stack))
    if  ndims != 3:
        raise ValueError(f'Stack must have exactly three dimensions, has instead {ndims}')
    if stack.dtype != np.float64:
        try:
            stack = np.float64(stack)
        except Exception:
            raise TypeError(f'Stack elements must be numpy.float64 or convertible to it, are instead {stack.dtype}')

    # Input check: max_threads must be an int between 1 and maximum available threads
    max_processes = cpu_count()
    if max_threads is None:
        max_threads = int(max_processes/2)
    else:
        if type(max_threads) is not int:
            try:
                max_threads = int(max_threads)
            except Exception:
                raise TypeError(f'max_threads must be of type int, is instead {type(max_threads)}')
            
        if max_threads <= 0 or max_threads >= max_processes:
            raise ValueError(f'max_threads must be between 1 and {max_processes-1} (inclusive), is instead {max_threads}')
    
    # Input check: calib must be a ndarray of float64 with same (sy,sx) as stack
    if calib is None:
        calib = np.zeros_like(stack[0])
    else:
        if type(calib) is not np.ndarray:
            raise TypeError(f'Calibration image must be a numpy.ndarray, is instead {type(calib)}')
        if calib.dtype != np.float64:
            try:
                calib = np.float64(calib)
            except Exception:
                raise TypeError(f'Calibration image elements must be numpy.float64 or convertible to it, are instead {calib.dtype}')
        if np.shape(calib) != np.shape(stack[0]):
            raise ValueError(f'Calib must have same shape as stack slices; instead, calib is {np.shape(calib)} while stack slices are {np.shape(stack[0])}')
        
    
    # Actual processing
    sz,sy,sx = np.shape(stack)
    idxs = range(sz)
    freeze_support()
    with ProcessPoolExecutor(max_workers=max_threads) as pool:
        uw_stack = list(pool.map(_unwrap_phase_scikit_single,stack,repeat(calib),idxs,repeat(sz)))

    return uw_stack

            
