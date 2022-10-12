import numpy as np
import pandas as pd
import uproot
import gc
from scipy import stats

class Cells:
    def __init__(self, infile, tree_name="ntuple",
             keys=["cell_et", "cell_phi", "cell_eta","metTruth_et", "metTruth_phi"], unit="GeV"):
        self.infile = infile
        self.keys = keys
        self.unit_scale = 0.001 if unit == "GeV" else 1
        self.events = uproot.open(infile+":"+tree_name)
        
    def _batch_cimg(self,data, phi_bins,  eta_bins,):
        all_events= list()
        for i in range(len(data['cell_phi'])):
            HB = stats.binned_statistic_2d(
                    data['cell_phi'][i], data['cell_eta'][i],
                    bins = [phi_bins, eta_bins],
                    values=[data["cell_ex"][i], data["cell_ey"][i]],
                    statistic='sum')
                    
            all_events += [HB.statistic.transpose([2,1,0])]
        return np.stack(all_events, axis=0)
                
    def cimg(self, x_bin, y_bin, batch_size=500):
        img_list = []
        for batch in self.events.iterate(step_size=batch_size, library="np"):
            _imgs =  self._batch_cimg(batch, x_bin,  y_bin)
            img_list +=[_imgs]
        gc.collect()
        #read all labels
        truth_met_grid = self.events.arrays(["metTruth_ex","metTruth_ey"],library="pd").to_numpy()*self.unit_scale
        return np.concatenate(img_list, axis=0)*self.unit_scale, truth_met_grid
        #np.concatenate(img_list, axis=0), np.concatenate(label_list, axis=0),
        
    def cimg_et(self, x_bin, y_bin, batch_size=500):
        img_list = []
        for batch in self.events.iterate(step_size=batch_size, library="np"):
            all_events= list()
            for i in range(len(batch['cell_phi'])):
                HB = stats.binned_statistic_2d(
                    batch['cell_phi'][i], batch['cell_eta'][i],
                    bins = [x_bin, y_bin],
                    values=batch["cell_et"][i],
                    statistic='sum')
                    
                all_events += [HB.statistic.transpose([1,0])]
                    
            img_list +=[np.stack(all_events, axis=0)]
            gc.collect()
        #read all labels
        truth_met_grid = self.events.arrays(["metTruth_et"],library="pd").to_numpy()*self.unit_scale
        return np.concatenate(img_list, axis=0)*self.unit_scale, truth_met_grid
    
    def cvector(self, x_bin, batch_size=500):
        phi_bins = x_bin
        img_list = list()
        for batch in self.events.iterate(step_size=batch_size, library="np"):
            all_events= list()
            for i in range(len(batch['cell_phi'])):
                HB = stats.binned_statistic(
                                batch['cell_phi'][i],
                    bins = phi_bins,
                    values=[batch["cell_ex"][i], batch["cell_ey"][i]],
                    statistic='sum')
                    
                all_events += [HB.statistic.transpose([1,0])]
            img_list +=[np.stack(all_events, axis=0)]
        truth_met_grid = self.events.arrays(["metTruth_ex","metTruth_ey"], library="pd").to_numpy()*self.unit_scale
        
        return np.concatenate(img_list, axis=0)*self.unit_scale, truth_met_grid #truth_met_grid.unstack(level=1).values # img and label

    def cvector_et(self, x_bin, batch_size=500):
        phi_bins = x_bin
        vec_list = list()
        v_label_lsit = list()
        for batch in self.events.iterate(step_size=batch_size, library="np"):
            all_events= list()
            for i in range(len(batch['cell_phi'])):
                HB = stats.binned_statistic(
                                batch['cell_phi'][i],
                    bins = phi_bins,
                    values=batch["cell_et"][i],
                    statistic='sum')
                    
                all_events += [HB.statistic]
            vec_list +=[np.stack(all_events, axis=0)]
        
            batch_label = list()
            for i in range(len(batch['metTruth_phi'])):
                HB = stats.binned_statistic(
                                        batch['metTruth_phi'][i],
                                        bins = phi_bins,
                                        values=batch["metTruth_et"][i],
                                        statistic='sum')
                batch_label += [HB.statistic]
            
            v_label_lsit +=[np.stack(batch_label, axis=0)]
        
        return np.concatenate(vec_list, axis=0)*self.unit_scale, np.concatenate(v_label_lsit, axis=0)*self.unit_scale #truth_met_grid.unstack(level=1).values # img and label
