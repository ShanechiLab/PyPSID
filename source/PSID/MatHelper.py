""" 
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

Helps with interfacing with matlab
"""

import scipy.io as sio
import numpy as np
import h5py

def loadmat(file_path, variable_names=None):
    "Loads a mat file as a dictionary"
    try:
        # mat_dict = sio.loadmat(file_path, matlab_compatible=True, variable_names=variable_names)
        mat_dict = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True, chars_as_strings=True, variable_names=variable_names)
    except NotImplementedError: # Runs for v7.3: 'Please use HDF reader for matlab v7.3 files'
        mat_dict = h5py.File(file_path)

    return _check_keys(mat_dict)

# From https://stackoverflow.com/a/8832212/2275605
def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
        elif isinstance(d[key], np.ndarray) and len(d[key]) > 0 and isinstance(d[key].item(0), sio.matlab.mio5_params.mat_struct):
            for i in range( d[key].size ):
                if isinstance(d[key].item(i), sio.matlab.mio5_params.mat_struct):
                    d[key].itemset( i, _todict( d[key].item(i) ) )
        else:
            pass

    return d

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for key in matobj._fieldnames:
        elem = matobj.__dict__[key]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(elem)
        elif isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem.item(0), sio.matlab.mio5_params.mat_struct):
            for i in range( elem.size ):
                if isinstance(elem.item(i), sio.matlab.mio5_params.mat_struct):
                    elem.itemset(i, _todict( elem.item(i) ) )
            d[key] = elem
        else:
            d[key] = elem
    return d
