
import os, sys
import numpy as np
import h5py
import argparse
import uproot
from cells import cell_data
from util import save_h5

def main():
    parser = argparse.ArgumentParser(
            prog = 'dump_clt_h5.py',
            description = 'Read data from root file then save the numpy arrays to h5 file.'
            )
            
    parser.add_argument('--input_file', help='root file')
    parser.add_argument('--output_dir', help='dir to save test,train files')
    parser.add_argument('--tree', default='ntuple', help='tree name')
    parser.add_argument('--unit', default=None,
                                help='GeV, convert to GeV.')
    
    args = parser.parse_args()
        
    #x_bin = np.linspace(-3.15, 3.15, num=args.phi_bins+1)
    #y_bin = np.linspace(-5, 5, num=args.eta_bins+1)
    
    unit_scale = 1.
    if args.unit == 'GeV':
        unit_scale=0.001
    
    group = ['clt_et', 'clt_ex', 'clt_ey', 'clt_phi', 'clt_eta']
    Nevents=0
    with uproot.open(args.input_file+":"+args.tree) as events:
        print(events.keys())
        batch_datasets = [] 
        for batch in events.iterate(step_size=1023, library="np"):
            batch_len = len(batch['clt_et'])
            for bl in range(batch_len):
                nvt = np.stack([batch['clt_et'][bl]*unit_scale,
                  batch['clt_ex'][bl]*unit_scale,
                  batch['clt_ey'][bl]*unit_scale,
                  batch['clt_phi'][bl],
                  batch['clt_eta'][bl]
                 ], axis=-2)
                 
                 
                label_group = ['metTruth_ex', 'metTruth_ex', 'metTruth_et']
                labels = np.array([batch[label_group[0]][bl],
                                batch[label_group[1]][bl],
                                batch[label_group[2]][bl] ]
                                )
                h5_train_dataset = {'clt':nvt,
                        'labels': labels
                     }
                save_h5(f"{args.output_dir}/train_{Nevents}.h5", h5_train_dataset)
                Nevents +=1
                

        #labels are just few numbers 
        #so read them all in ones
        
   
    
if __name__ == '__main__':
    main()
