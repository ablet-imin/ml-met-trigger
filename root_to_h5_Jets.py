
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
            prog = 'root_to_h5_Jets.py',
            description = 'Read data from root file then save the numpy arrays to h5 file.'
            )
            
    parser.add_argument('--input_file', help='root file')
    parser.add_argument('--output', help='dir to save test,train files')
    parser.add_argument('--tree', default='ntuple', help='tree name')
    parser.add_argument('--unit', default=None,
                                help='GeV, convert to GeV.')
    #parser.add_argument('--phi_bins', type=int, default=64, help='number of bins for phi')
    #parser.add_argument('--eta_bins', type=int, default=50, help='number of bins for eta')
    parser.add_argument('--jet_name', default="AntiKt4emtopoCalo420Jets", help=" Jet container name")
                                
    args = parser.parse_args()
        
    #x_bin = np.linspace(-3.15, 3.15, num=args.phi_bins+1)
    #y_bin = np.linspace(-5, 5, num=args.eta_bins+1)

    unit_scale = 0.001 if args.unit == "GeV" else 1.
    
    METS = ["MET_Calo_pt", "MET_Calo_px", "MET_Calo_py", "MET_Calo_phi",
                "met_truth_pt", "met_truth_px", "met_truth_py","met_truth_phi",
                "pufitCalo422_pt","pufitCalo422_px", "pufitCalo422_py","pufitCalo422_phi",
                 "pufitCalo422SK_pt", "pufitCalo422SK_px", "pufitCalo422SK_py",
                 "pufitCalo422SK_phi"]
    jet_name = args.jet_name
    jet_vars = ['pt', 'eta', 'phi',]
    expressions= [f'{jet_name}_{var}' for var in jet_vars]
    expressions= expressions + METS # include met branches
    
    NUMBER_OF_JETS = 3 #len(batch[f"{jet_name}_{jet_vars[0]}"])
    
    _labels = {}

    for met in METS:
        _labels[met] = []
    tree  = args.input_file+":"+args.tree
    
    Jet_pt_list = list()
    Jet_eta_list = list()
    Jet_phi_list = list()
    for batch in uproot.iterate(tree, expressions=expressions,
                                step_size='1 GB', library="np"):
        
        #Loop over events
        for i in range(len(batch[f"{jet_name}_{jet_vars[0]}"])):
            #each event have many jets.
            #batch[var][i] is the ith events and all jets
            #[:NUMBER_OF_JETS] select first NUMBER_OF_JETS jets
            #3 var and 3 jets --> 3x3 matrics for each event
            Jet_pt_list +=[batch[f"{jet_name}_pt"][i][:NUMBER_OF_JETS]]
            Jet_eta_list +=[batch[f"{jet_name}_eta"][i][:NUMBER_OF_JETS]]
            Jet_phi_list +=[batch[f"{jet_name}_phi"][i][:NUMBER_OF_JETS]]
            
        # read all labels
        for met in METS:
            _labels[met] = _labels[met] + [batch[met] * unit_scale]
        
        gc.collect()

                     
    #hdt = h5py.special_dtype(vlen=np.float32)
    h5f = h5py.File(f"{args.output}", 'w')
    Jet_pts = np.array(Jet_pt_list)
    Jet_etas = np.array(Jet_eta_list)
    Jet_phis = np.array(Jet_phi_list)
    
    print("Jet pt shapes:", Jet_pts.shape)
    h5f.create_dataset('jet_pt', data = Jet_pts)
    h5f.create_dataset('jet_eta', data = Jet_etas)
    h5f.create_dataset('jet_phi', data = Jet_phis)
    
    for met in METS:
        _data = np.concatenate(_labels[met], axis=0)
        h5f.create_dataset(met, data=_data)
    h5f.close()
    
    
if __name__ == '__main__':
    main()
    
