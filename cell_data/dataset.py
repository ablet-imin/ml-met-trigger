import os,sys
import h5py
import numpy as np

def _get_data(filepath, labelkey):
    with h5py.File(filepath, 'r') as h5f:
        et  = np.array(h5f['et'])
        ex  = np.array(h5f['ex'])
        ey  = np.array(h5f['ey'])
        eta  = np.array(h5f['eta'])
        phi  = np.array(h5f['phi'])
        noise = np.array(h5f['noise'])
        label = np.array(h5f[labelkey])
        #noise = et/noise
        eta = np.nan_to_num(eta)
        phi = np.nan_to_num(phi)
    return et, ex, ey, noise,  phi, eta, label
    

def _get_data_all(filepath, labelkey):
    et, ex, ey, eta, phi, noise = [],[],[],[],[],[]
    label=[]
    for i, file in enumerate(filepath):
        with h5py.File(file, 'r') as h5f:
            et += [np.array(h5f['et'])]
            ex += [np.array(h5f['ex'])]
            ey += [np.array(h5f['ey'])]
            eta += [np.nan_to_num(np.array(h5f['eta']))]
            phi += [np.nan_to_num(np.array(h5f['phi']))]
            noise += [np.array(h5f['noise'])]
            label += [np.array(h5f[labelkey])]
    return (np.concatenate(et, axis=0),
            np.concatenate(ex, axis=0),
            np.concatenate(ey, axis=0),
            np.concatenate(noise, axis=0),
            np.concatenate(phi, axis=0),
            np.concatenate(eta, axis=0),
            np.concatenate(label, axis=0) )

def get_data(filepath, labelkey):
    if isinstance(filepath, list):
        return _get_data_all(filepath, labelkey)
    else:
        return _get_data(filepath, labelkey)
        


