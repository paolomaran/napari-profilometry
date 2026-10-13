import h5py
import numpy as np
import pytest


@pytest.fixture
def sample_h5(tmp_path):
    '''
    Creates a sample HDF5 file with two datasets and measurement metadata.
    '''

    path = tmp_path / "test_measurement.h5"

    data0 = np.arange(3*6*8,dtype=np.uint16).reshape(3,6,8)
    data1 = np.arange(2*6*8,dtype=np.uint16).reshape(2,6,8)

    with h5py.File(path,'w') as f:
        measurement = f.create_group('measurement')
        experiment = measurement.create_group('test_experiment')

        experiment.attrs['magnification'] = 20.0
        experiment.attrs['pixelsize'] = 0.5
        experiment.attrs['wavelength'] = 0.532

        experiment.create_dataset('t0000/frames',data=data0)
        experiment.create_dataset('t0001/frames',data=data1)

    return path