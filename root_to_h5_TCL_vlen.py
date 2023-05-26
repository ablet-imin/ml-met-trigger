
import os, sys
import numpy as np
import h5py
import argparse
import uproot as uproot
from util import save_h5
#from scipy import stats
import gc
import sys

def main():
    parser = argparse.ArgumentParser(
            prog = 'root_to_h5_TCL.py',
            description = 'Read data from root file then save the numpy arrays to h5 file.'
            )
            
    parser.add_argument('--input_file', help='root file')
    parser.add_argument('--output', help='dir to save test,train files')
    parser.add_argument('--tree', default='ntuple', help='tree name')
    parser.add_argument('--unit', default=None,
                                help='GeV, convert to GeV.')
    #parser.add_argument('--phi_bins', type=int, default=64, help='number of bins for phi')
    #parser.add_argument('--eta_bins', type=int, default=50, help='number of bins for eta')
    parser.add_argument('--tcl_name', default="Calo422TopoClusters", help="name tcl container")
                                
    args = parser.parse_args()
        
    #x_bin = np.linspace(-3.15, 3.15, num=args.phi_bins+1)
    #y_bin = np.linspace(-5, 5, num=args.eta_bins+1)

    unit_scale = 0.001 if args.unit == "GeV" else 1.
    
    METS = ["MET_Calo_pt", "MET_Calo_px", "MET_Calo_py",
                "met_truth_pt", "met_truth_px", "met_truth_py",
                "pufitCalo422_pt","pufitCalo422_px", "pufitCalo422_py", "pufitCalo422SK_pt",
                "pufitCalo422SK_px", "pufitCalo422SK_py"]
    cluster_name = args.tcl_name
    cl_vars = ['et', 'eta', 'phi', 'n_cell', 'significance',
                'eng_frac_max', 'eng_frac_em', 'cell_sig']
    expressions= [f'{cluster_name}_{var}' for var in cl_vars]
    expressions= expressions + METS # include met branches
    
    _labels = {}

    for met in METS:
        _labels[met] = []
    tree  = args.input_file+":"+args.tree
    
    events_list = list()
    for batch in uproot.iterate(tree, expressions=expressions,
                                step_size='1 GB', library="np"):
        
        for i in range(len(batch[f"{cluster_name}_{cl_vars[0]}"])):
            event_array = list()
            for var in cl_vars:
                event_array += [batch[f"{cluster_name}_{var}"][i]]
            events_list += [np.stack(event_array, axis=0).flatten()]
        
            
        # read all labels
        for met in METS:
            _labels[met] = _labels[met] + [batch[met] * unit_scale]
        
        gc.collect()

                     
    hdt = h5py.special_dtype(vlen=np.float32)
    h5f = h5py.File(f"{args.output}", 'w')
    cl_data = np.array(events_list, dtype=object)
    h5f.create_dataset('tcl', data = cl_data, dtype=hdt)
    
    #dt = h5py.special_dtype(vlen=str)
    #h5f.create_dataset('cl_vars', cl_vars, dtype=dt)
    for met in METS:
        _data = np.concatenate(_labels[met], axis=0)
        h5f.create_dataset(met, data=_data)
    h5f.close()
    
    
if __name__ == '__main__':
    main()
    
