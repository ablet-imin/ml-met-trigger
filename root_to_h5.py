
import os, sys
import numpy as np
import h5py
import argparse
import uproot
from cells import cell_data
from util import save_h5

def main():
    parser = argparse.ArgumentParser(
            prog = 'root_to_h5.py',
            description = 'Read data from root file then save the numpy arrays to h5 file.'
            )
            
    parser.add_argument('--input_file', help='root file')
    parser.add_argument('--output', help='h5 file name')
    parser.add_argument('--tree', default='ntuple', help='tree name')
    parser.add_argument('--stat', type=str, choices=['sum', 'mean'],
                                default='sum', help='tree name')
    parser.add_argument('--var', type=str,
                                default='cell_et', help='branch name in the tree')
    parser.add_argument('--label', type=str,
                                default='metTruth_et', help='branch name in the tree')
    parser.add_argument('--phi_bins', type=int, default=32, help='number of bins for phi')
    parser.add_argument('--eta_bins', type=int, default=50, help='number of bins for eta')
    
    args = parser.parse_args()
        
    x_bin = np.linspace(-3.15, 3.15, num=args.phi_bins+1)
    y_bin = np.linspace(-5, 5, num=args.eta_bins+1)
    
    with uproot.open(args.input_file+":"+args.tree) as events:
        print(events.keys())
        images, labels = cell_data(events, x_bin,
                                        y_bin, weight=args.var,
                                        label=args.label,
                                         statistic=args.stat, batch_size=1000)
    
    print(f"Cell image shape: {images.shape}")
    print(f"Image label shape: {labels.shape}")
    
    from sklearn.model_selection import train_test_split
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels,
                                        test_size=0.2, random_state=42,
                                        shuffle=True
                                        )
    
    h5_train_dataset = {'images':x_train,
                      'labels': y_train
                     }
    save_h5(f"data/train_{args.output}", h5_train_dataset)
    
    h5_test_dataset = {'images':x_test,
                      'labels': y_test
                     }
    save_h5(f"data/test_{args.output}", h5_test_dataset)
    
    
if __name__ == '__main__':
    main()
