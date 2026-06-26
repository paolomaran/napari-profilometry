'''
Simple processor to force continuity (within a certain range) of an image stack.

Author: Paolo Maran (Polimi)
'''

import numpy as np 


def force_continuity_stack(stack: np.ndarray,roi: tuple,set_zero_first_image: bool = False):
    '''
    Takes a stack of profilometry images and forces the average difference
    in the region roi to be between -pi and +pi.

    Args
    --------
    stack: ndarray (sz x sy x sx)
        The image array containing the wrapped phase.
        Must be ordered as (z, y, x).
        Element type must be float.
    roi: 
        (ymin,ymax,xmin,xmax). A tuple of 4 ints representing the four corners of the 
        rectangular ROI used for continuity evaluation.
    set_zero_first_image: bool (Optional, default: False)
        if True, the average over roi of stack[0] will be subtracted to the whole stack.
        In this way, the average height over stack will be equal to 0 for slice 0.

    
    Returns
    --------
    stack_cont: ndarray (sz x sy x sx)
        A numpy array with same dimensions as stack which is continuous in the roi given.

    Raises
    --------
    ValueError
        If roi is invalid.
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


    #Input check: roi must have four elements, and the minimum of the roi must be smaller than the maximum
    #both in x and y directions.
    try:
        xmin = roi[2]
        xmax = roi[3]
        ymin = roi[0]
        ymax = roi[1]
    except Exception:
        raise ValueError(f'roi must be a sequence of four ints, but is instead {roi}')
    
    if xmin>=xmax:
        raise ValueError(f'xmin must be smaller than xmax: xmin = {xmin}, xmax = {xmax}')
    if ymin >= ymax: 
        raise ValueError(f'ymin must be smaller than ymax: ymin = {ymin}, ymax = {ymax}')

       

    raw = stack.copy()

    if set_zero_first_image:
        raw = raw-np.average(raw[0,ymin:ymax,xmin:xmax])
    
    (sz,sy,sx) = np.shape(raw)


    for i in range(sz-1):
        roi_curr = raw[i+1,ymin:ymax,xmin:xmax] - raw[i,ymin:ymax,xmin:xmax]
        diff = np.average(roi_curr) + np.pi
        if diff < 0 or diff > 2*np.pi:
            if diff<0:
                raw[i+1] -= (diff//(2*np.pi) )*2*np.pi
            else:
                raw[i+1] -= (diff//(2*np.pi) )*2*np.pi

    return raw

