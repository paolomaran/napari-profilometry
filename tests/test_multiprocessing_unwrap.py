
import importlib
import numpy as np
import pytest


@pytest.fixture(params=[
    "PUMA_multiprocessing_unwrap",
    "SCIKIT_multiprocessing_unwrap",
])
def processor_module(request,monkeypatch):
    '''
    Provides both multiprocessing implementations
    for the same collection of validation tests.
    '''

    module = importlib.import_module(
        "napari_profilometry.unwrap_processors."+request.param
    )

    monkeypatch.setattr(module,"cpu_count",lambda: 8)
    return module


def run_processor(module,stack,**kwargs):
    '''
    Calls the multiprocessing function corresponding
    to the supplied module.
    '''

    if module.__name__.endswith("PUMA_multiprocessing_unwrap"):
        return module.multiPUMA_unwrap_processor(stack,**kwargs)

    return module.multiSCIKIT_unwrap_processor(stack,**kwargs)


def test_multiprocessing_invalid_type(processor_module):
    '''
    Tests that the processor rejects input
    that is not a numpy.ndarray.
    '''

    data = [[[0.0]*16]*16]*3

    with pytest.raises(TypeError,match="numpy.ndarray"):
        run_processor(processor_module,data,max_threads=1)


def test_multiprocessing_invalid_dimensions(processor_module):
    '''
    Tests that the processor requires exactly
    three dimensions.
    '''

    data = np.zeros((16,16))

    with pytest.raises(ValueError,match="exactly three dimensions"):
        run_processor(processor_module,data,max_threads=1)


def test_multiprocessing_invalid_dtype(processor_module):
    '''
    Tests that non-numeric stack elements
    raise a TypeError.
    '''

    data = np.full((3,16,16),'invalid')

    with pytest.raises(TypeError,match="convertible"):
        run_processor(processor_module,data,max_threads=1)


def test_multiprocessing_invalid_threads_type(processor_module):
    '''
    Tests that an invalid process count
    raises a TypeError.
    '''

    data = np.zeros((3,16,16))

    with pytest.raises(TypeError,match="max_threads must be"):
        run_processor(processor_module,data,max_threads="invalid")


@pytest.mark.parametrize("threads",[0,-1,8,9])
def test_multiprocessing_invalid_threads_count(processor_module,threads):
    '''
    Tests that the process count is greater than zero
    and smaller than the available CPU count.
    '''

    data = np.zeros((3,16,16))

    with pytest.raises(ValueError,match="max_threads must be between"):
        run_processor(processor_module,data,max_threads=threads)


def test_multiprocessing_invalid_calibration_type(processor_module):
    '''
    Tests that calibration data must be a numpy.ndarray.
    '''

    data = np.zeros((3,16,16))

    with pytest.raises(TypeError,match="Calibration image must be"):
        run_processor(processor_module,data,max_threads=1,calib=[[0.0]*16]*16)


def test_multiprocessing_invalid_calibration_shape(processor_module):
    '''
    Tests that the calibration image matches
    the spatial dimensions of the stack.
    '''

    data = np.zeros((3,16,16))
    calib = np.zeros((8,8))

    with pytest.raises(ValueError,match="Calib must have same shape"):
        run_processor(processor_module,data,max_threads=1,calib=calib)


def test_multiprocessing_invalid_calibration_dtype(processor_module):
    '''
    Tests that non-numeric calibration data
    raises a TypeError.
    '''

    data = np.zeros((3,16,16))
    calib = np.full((16,16),'invalid')

    with pytest.raises(TypeError,match="Calibration image elements"):
        run_processor(processor_module,data,max_threads=1,calib=calib)


'''
-------------------------------------------------------------
           Multiprocessing execution path tests 
-------------------------------------------------------------
'''


class SynchronousExecutor:
    '''
    Replacement for ProcessPoolExecutor that executes
    work synchronously during testing.
    '''

    def __init__(self,max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        return False

    def map(self,func,*iterables):
        return map(func,*iterables)


def test_multiprocessing_processing(processor_module,monkeypatch):
    '''
    Tests multiprocessing stack processing without creating
    actual subprocesses.
    Tests include:
    - checking that the output shape is correct
    - checking that all frames are processed
    - checking that calibration is applied correctly
    '''

    module = processor_module
    monkeypatch.setattr(module,"ProcessPoolExecutor",SynchronousExecutor)

    # Substitute both algorithms with a predictable operation
    if module.__name__.endswith("PUMA_multiprocessing_unwrap"):
        monkeypatch.setattr(module,"unwrap_single_image",lambda image: image+1.0)
    else:
        monkeypatch.setattr(module.scikit,"unwrap_phase",lambda image: image+1.0)

    data = np.arange(3*8*8,dtype=np.float64).reshape(3,8,8)
    calib = np.ones((8,8))*0.5

    result = run_processor(module,data,max_threads=2,calib=calib)

    assert result.shape == (3,8,8)
    np.testing.assert_array_equal(result,data+0.5)


def test_multiprocessing_default_threads(processor_module,monkeypatch):
    '''
    Tests that the default number of processes
    is half the available CPU count.
    '''

    module = processor_module
    used_threads = []

    class TrackingExecutor(SynchronousExecutor):
        def __init__(self,max_workers=None):
            super().__init__(max_workers)
            used_threads.append(max_workers)

    monkeypatch.setattr(module,"ProcessPoolExecutor",TrackingExecutor)

    if module.__name__.endswith("PUMA_multiprocessing_unwrap"):
        monkeypatch.setattr(module,"unwrap_single_image",lambda image: image)
    else:
        monkeypatch.setattr(module.scikit,"unwrap_phase",lambda image: image)

    data = np.zeros((2,8,8))
    run_processor(module,data)

    assert used_threads == [4]


def test_multiprocessing_thread_conversion(processor_module,monkeypatch):
    '''
    Tests that a numeric string specifying the process
    count is converted to an integer.
    '''

    module = processor_module
    used_threads = []

    class TrackingExecutor(SynchronousExecutor):
        def __init__(self,max_workers=None):
            super().__init__(max_workers)
            used_threads.append(max_workers)

    monkeypatch.setattr(module,"ProcessPoolExecutor",TrackingExecutor)

    if module.__name__.endswith("PUMA_multiprocessing_unwrap"):
        monkeypatch.setattr(module,"unwrap_single_image",lambda image: image)
    else:
        monkeypatch.setattr(module.scikit,"unwrap_phase",lambda image: image)

    data = np.zeros((2,8,8))
    run_processor(module,data,max_threads="2")

    assert used_threads == [2]


def test_multiprocessing_input_conversion(processor_module,monkeypatch):
    '''
    Tests that integer stacks and calibration images
    are converted to float64 during processing.
    '''

    module = processor_module
    monkeypatch.setattr(module,"ProcessPoolExecutor",SynchronousExecutor)

    if module.__name__.endswith("PUMA_multiprocessing_unwrap"):
        monkeypatch.setattr(module,"unwrap_single_image",lambda image: image)
    else:
        monkeypatch.setattr(module.scikit,"unwrap_phase",lambda image: image)

    data = np.ones((3,8,8),dtype=np.uint16)*5
    calib = np.ones((8,8),dtype=np.uint16)*2

    result = run_processor(module,data,max_threads=1,calib=calib)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result,np.ones((3,8,8))*3)


'''
-------------------------------------------------------------
                Actual integration tests 
-------------------------------------------------------------
'''

def test_scikit_real_multiprocessing():
    '''
    Tests SCIKIT multiprocessing using real subprocesses.
    Tests include:
    - checking that multiple frames are processed
    - checking that the output shape is correct
    - checking the numerical agreement with single-image unwrapping
    '''

    from multiprocessing import cpu_count
    from napari_profilometry.unwrap_processors.SCIKIT_multiprocessing_unwrap import multiSCIKIT_unwrap_processor
    from napari_profilometry.unwrap_processors.SCIKIT_unwrap import unwrap_phase_scikit

    if cpu_count() <= 1:
        pytest.skip("At least two CPUs are required by the current implementation")

    y,x = np.mgrid[0:16,0:16]
    phase = 0.4*x + 0.2*y
    wrapped = np.angle(np.exp(1j*phase))
    data = np.stack([wrapped,wrapped+0.1,wrapped+0.2])

    result = multiSCIKIT_unwrap_processor(data,max_threads=1)
    expected = np.stack([unwrap_phase_scikit(frame) for frame in data])

    assert result.shape == data.shape
    np.testing.assert_allclose(result,expected,atol=1e-10)
