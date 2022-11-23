import numpy as np
import pandas as pd
import uproot
import gc
from scipy import stats


def cell_data(events,phi_bins, eta_bins, weight,
                statistic='sum', unit="GeV", batch_size=1000):
    unit_scale = 0.001 if unit == "GeV" else 1
    def _batch_cimg( data):
        all_events = list()
        for i in range(len(data['cell_phi'])):
            HB = stats.binned_statistic_2d(
                data['cell_eta'][i], data['cell_phi'][i],
                bins=[eta_bins,phi_bins],
                values=data[weight][i],
                statistic=statistic)
            all_events += [HB.statistic]
        return np.stack(all_events, axis=0)
        
    img_list = []
    for batch in events.iterate(step_size=batch_size, library="np"):
            _imgs = _batch_cimg(batch)
            img_list += [_imgs]
    gc.collect()
    
    # read all labels
    truth_met_grid = events.arrays(
        ["metTruth_ex", "metTruth_ey"], library="pd").to_numpy() * unit_scale
    return np.concatenate(img_list, axis=0) * \
        unit_scale, truth_met_grid

