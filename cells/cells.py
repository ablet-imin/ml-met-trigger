import numpy as np
import pandas as pd
import uproot
import gc
from scipy import stats


class Cells:
    def __init__(
            self,
            infile,
            tree_name="ntuple",
            keys=[
                "cell_et",
                "cell_phi",
                "cell_eta",
                "metTruth_et",
                "metTruth_phi"],
            unit="GeV"):
        self.infile = infile
        self.keys = keys
        self.unit_scale = 0.001 if unit == "GeV" else 1
        self.events = uproot.open(infile + ":" + tree_name)

    def _batch_cimg(self, data, phi_bins, eta_bins,):
        all_events = list()
        for i in range(len(data['cell_phi'])):
            HB = stats.binned_statistic_2d(
                data['cell_phi'][i], data['cell_eta'][i],
                bins=[phi_bins, eta_bins],
                values=[data["cell_ex"][i], data["cell_ey"][i]],
                statistic='sum')

            all_events += [HB.statistic.transpose([2, 1, 0])]
        return np.stack(all_events, axis=0)

    def cimg(self, x_bin, y_bin, batch_size=500):
        img_list = []
        for batch in self.events.iterate(step_size=batch_size, library="np"):
            _imgs = self._batch_cimg(batch, x_bin, y_bin)
            img_list += [_imgs]
        gc.collect()
        # read all labels
        truth_met_grid = self.events.arrays(
            ["metTruth_ex", "metTruth_ey"], library="pd").to_numpy() * self.unit_scale
        return np.concatenate(img_list, axis=0) * \
            self.unit_scale, truth_met_grid
        #np.concatenate(img_list, axis=0), np.concatenate(label_list, axis=0),

    def cimg_et(self, x_bin, y_bin, batch_size=500):
        _X, _Y = self.cimg(x_bin, y_bin, batch_size)
        _X = np.square(_X).sum(axis=-1 )
        _X = np.sqrt(_X)
        gc.collect()
        
        _Y = np.square(_Y).sum(axis=-1 )
        _Y = np.sqrt(_Y)
        gc.collect()
        
        return _X, _Y

