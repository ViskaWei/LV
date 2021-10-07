import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .dnn import DNN 
from lv.constants import Constants as c


class DNNPipeline(object):
    def __init__(self):
        self.dnn = DNN()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None

    def load_data(self, Rg, W, N=10000):
        W = c.Ws[W]
        nn= N // 1000
        DATA_PATH = f"/scratch/ceph/swei20/data/dnn/{Rg}/rbf_R{W[2]}_{nn}k.h5"
        PC_PATH = f"/scratch/ceph/swei20/data/dnn/pc/bosz_{W[3]}_R{W[2]}.h5"
        with h5py.File(DATA_PATH, 'r') as f:
            flux = f[f'normflux'][()]
            pnorm=f['pnorm'][()]
            pval =f['pval'][()]
            wave=f['wave'][()]
        print(flux.shape, pnorm.shape,wave.shape)
        flux0, wave0 = self.get_flux_in_Wrange(flux, wave, W)   

    def get_flux_in_Wrange(self, flux, wave, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]