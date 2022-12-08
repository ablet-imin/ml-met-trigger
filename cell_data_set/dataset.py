import os,sys
import h5py
import numpy as np

def load_data(filepath, datakey, labelkey):
    with h5py.File(filepath, 'r') as h5f:
        data  = np.array(h5f[datakey])
        label = np.array(h5f[labelkey])
    return data, label
    

def get_dataset (h5_path, set_typ):
    #read ex
    def _load(h5_file):
        return load_data(h5_file,
            datakey="images",
            labelkey="labels")
        
    ex, label_ex = _load(h5_path+f"/ex/{set_typ}.h5")
    
    #read ey
    ey, label_ey = _load(h5_path+f"/ey/{set_typ}.h5")
    
    #read phi
    phi, _ = _load(h5_path+f"/phi/{set_typ}.h5")
    
    #read eta
    eta, _ = _load(h5_path+f"/eta/{set_typ}.h5")
    
    #stack all elements
    _data = np.stack([ex, ey, eta, phi], axis=-1)
    _label = np.stack([label_ex, label_ey], axis=-1)
    
    return _data, _label
    
def get_et(h5_path, set_typ):
    et, label_et = load_data(h5_path+f"/et/{set_typ}.h5",
                            datakey="images",
                            labelkey="labels")
    return et, label_et

def get_train_et(h5_path):
    return get_et(h5_path, set_typ='train')
    
def get_test_et(h5_path):
    return get_et(h5_path, set_typ='test')
    
    
def get_train(h5_path):
    return get_dataset (h5_path, set_typ='train')

def get_test(h5_path):
    return get_dataset (h5_path, set_typ='test')

