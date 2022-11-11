import os, sys
import numpy as np
import h5py

def save_h5(out_file, dataset:dict):
    h5f = h5py.File(out_file, 'w')
    for data_name, data_values in dataset.items():
        h5f.create_dataset(data_name, data=data_values)
    h5f.close()

def load_data(filepath, datakey, labelkey):
    with h5py.File(filepath, 'r') as h5f:
        data  = np.array(h5f[datakey])
        label = np.array(h5f[labelkey])
    return data, label
    
