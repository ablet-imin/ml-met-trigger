
import os, sys
import numpy as np
import h5py
import argparse
import uproot as uproot
from scipy import stats
import gc
import sys

def save_h5(out_file, dataset:dict):
    h5f = h5py.File(out_file, 'w')
    for data_name, data_values in dataset.items():
        h5f.create_dataset(data_name, data=data_values)
    h5f.close()
    
def main():
    parser = argparse.ArgumentParser(
            prog = 'root_to_h5.py',
            description = 'Read data from root file then save the numpy arrays to h5 file.'
            )
            
    parser.add_argument('--input_file', help='root file')
    parser.add_argument('--output', help='dir to save test,train files')
    parser.add_argument('--tree', default='ntuple', help='tree name')
    parser.add_argument('--unit', default=None,
                                help='GeV, convert to GeV.')
    parser.add_argument('--phi_bins', type=int, default=64, help='number of bins for phi')
    parser.add_argument('--eta_bins', type=int, default=50, help='number of bins for eta')
    
    args = parser.parse_args()
        
    x_bin = np.linspace(-3.15, 3.15, num=args.phi_bins+1)
    y_bin = np.linspace(-5, 5, num=args.eta_bins+1)
    
    def _batch_cimg( data, phi_bins, eta_bins, weight,
                    eta='cells_eta', phi='cells_phi',
                    statistic='sum'):
        all_events = list()
        for i in range(len(data[phi])):
            HB = stats.binned_statistic_2d(
                data[eta][i], data[phi][i],
                bins=[eta_bins,phi_bins],
                values=data[weight][i],
                statistic=statistic)
            all_events += [HB.statistic]
        print('batch complete....')
        return np.stack(all_events, axis=0)
    
    
    unit_scale = 0.001 if args.unit == "GeV" else 1.
    
    METS = ["MET_Calo_pt", "MET_Calo_px", "MET_Calo_py",
                "met_truth_pt", "met_truth_px", "met_truth_py",
                "pufitCalo422_pt","pufitCalo422_px", "pufitCalo422_py", "pufitCalo422SK_pt",
                "pufitCalo422SK_px", "pufitCalo422SK_py"]
                
    expressions=["cells_et", "cells_ex", "cells_ey",
     "cells_eta", "cells_phi", "cells_totalNoise"]
    expressions= expressions + METS # include met branches
    
    et_list = []
    ex_list = []
    ey_list = []
    eta_list = []
    phi_list = []
    noise_list = []
    _labels = {}
    for met in METS:
        _labels[met] = []
    tree  = args.input_file+":"+args.tree
    for batch in uproot.iterate(tree, expressions=expressions,
                                step_size='1 GB', library="np"):
        
        #get ex
        _ex = _batch_cimg(batch, x_bin, y_bin,
                            weight="cells_ex",
                            statistic='sum')
        ex_list += [_ex]
        
        #get ey
        _ey = _batch_cimg(batch, x_bin, y_bin,
                            weight="cells_ey",
                            statistic='sum')
        ey_list += [_ey]
        
        #get et
        _et = np.sqrt(_ex*_ex+_ey*_ey)
        #_et = _batch_cimg(batch, x_bin, y_bin,
        #                    weight="cells_et",
        #                    statistic='sum')
        et_list += [_et]
        gc.collect()
        
        #get phi
        _phi = _batch_cimg(batch, x_bin, y_bin,
                            weight="cells_phi",
                            statistic='mean')
        phi_list += [_phi]
        
        #get eta
        _eta = _batch_cimg(batch, x_bin, y_bin,
                            weight="cells_eta",
                            statistic='mean')
        eta_list += [_eta]
        
        #get noise
        _noise = _batch_cimg(batch, x_bin, y_bin,
                            weight="cells_totalNoise",
                            statistic='sum')
        noise_list += [_noise]
        
        
        # read all labels
        for met in METS:
            _labels[met] = _labels[met] + [batch[met] * unit_scale]
        
        gc.collect()
        
    
    images_et  = np.concatenate(et_list, axis=0)
    images_ex  = np.concatenate(ex_list, axis=0)
    images_ey  = np.concatenate(ey_list, axis=0)
    images_eta  = np.concatenate(eta_list, axis=0)
    images_phi  = np.concatenate(phi_list, axis=0)
    images_noise  = np.concatenate(noise_list, axis=0)
    
    print(f"et shape: {images_et.shape}")
    h5_train_dataset = {'et': images_et,
                        'ex': images_ex,
                        'ey': images_ey,
                        'phi': images_phi,
                        'eta': images_eta,
                        'noise': images_noise
                     }
                     
    del et_list, ex_list, ey_list, eta_list,phi_list,noise_list
    
    for met in METS:
        h5_train_dataset[met] = np.concatenate(_labels[met], axis=0)
                     
    save_h5(f"{args.output}", h5_train_dataset)
        
    
if __name__ == '__main__':
    main()
