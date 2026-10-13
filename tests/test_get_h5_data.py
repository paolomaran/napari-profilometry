from napari_profilometry._get_h5_data import (
    get_h5_dataset,
    get_multiple_h5_datasets,
    get_datasets_index_by_name,
    get_h5_attr,
    get_group_name,
)


def test_get_h5_dataset(sample_h5):
    '''
    Tests loading a single dataset from an HDF5 file.
    '''

    import numpy as np

    data,found = get_h5_dataset(sample_h5,dataset_index=0)

    assert found == 2
    assert data.shape == (3,6,8)
    assert data.dtype == np.float32


def test_get_multiple_h5_datasets(sample_h5):
    '''
    Tests loading multiple datasets from an HDF5 file.
    Tests include:
    - checking that the number of datasets is correct
    - checking that datasets are cropped to the minimum frame count
    '''

    import numpy as np

    data,found = get_multiple_h5_datasets(sample_h5,[0,1])

    assert found == 2
    assert data.shape == (2,2,6,8)

    expected = np.arange(3*6*8).reshape(3,6,8)
    np.testing.assert_array_equal(data[0],expected[:2])


def test_get_datasets_index_by_name(sample_h5):
    '''
    Tests searching for datasets using their HDF5 path.
    '''

    indices,names = get_datasets_index_by_name(sample_h5,match='/t0000/')

    assert len(indices) == 1
    assert len(names) == 1
    assert '/t0000/' in names[0]


def test_get_h5_attr(sample_h5):
    '''
    Tests retrieving measurement attributes from an HDF5 file.
    '''

    magnification = get_h5_attr(sample_h5,'magnification')
    pixelsize = get_h5_attr(sample_h5,'pixelsize')
    wavelength = get_h5_attr(sample_h5,'wavelength')

    assert magnification == [20.0]
    assert pixelsize == [0.5]
    assert wavelength == [0.532]


def test_get_h5_attr_missing(sample_h5):
    '''
    Tests retrieving an attribute that does not exist.
    '''

    result = get_h5_attr(sample_h5,'nonexistent_attribute')

    assert result == []


def test_get_group_name(sample_h5):
    '''
    Tests retrieving measurement group names.
    '''

    groups,found = get_group_name(sample_h5,group_key='measurement')

    assert found == 1
    assert groups == ['/measurement/test_experiment']


def test_invalid_dataset_index(sample_h5):
    '''
    Tests that requesting a nonexistent dataset raises an error.
    '''

    import pytest

    with pytest.raises(AssertionError):
        get_h5_dataset(sample_h5,dataset_index=100)