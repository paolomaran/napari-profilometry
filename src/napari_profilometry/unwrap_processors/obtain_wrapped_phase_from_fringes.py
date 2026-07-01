'''
Simple function to obtain wrapped fringe images from 3 or more images of fringes.

Author: Paolo Maran, Politecnico di Milano
'''

import numpy as np

def obtain_wrapped_phase(image:np.ndarray):

    '''
    Processor that, given an image of a wrapped phase maps,
    unwraps each image using the PUMA algorithm.
    Due to very high computational demand and long execution time, this
    function should be called always in a separate thread.

    Parameters
    ----------
    wrapped_phase: ndarray (sp x sy x sx)
        The image stack containing the frignes images
        Must be ordered as sp,sy,sx, and must have exactly 3 dimensions.
        There must be at least 3 fringes images.
    Returns
    -------
    wrapped_phase: ndarray (sy x sx)
        Image stack of same dimensions as stack, containing the wrapped phase.
    '''

    
    # Input check: check that image is of correct type and shape
    if type(image) is not np.ndarray:
        raise TypeError(f'Image must be a numpy.ndarray, is instead {type(wrapped_phase)}')
    ndims = len(np.shape(image))
    if  ndims != 3:
        raise ValueError(f'Image must have exactly three dimensions, has instead {ndims}')
    sp,_,_ = np.shape(image)
    if sp<3:
        raise ValueError(f'There must be at least three images of fringes, instead {sp} were given.')

    image = image.astype(np.float64)
    _phase = np.zeros_like(image,dtype = np.float64)
        
    for i in range(sp):
        _phase += image[i]*np.exp(+1j*i*2*np.pi/sp)
    wrapped_phase = np.angle(_phase)

    return wrapped_phase