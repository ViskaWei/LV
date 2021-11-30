import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

from lv.util.constants import Constants as C
from lv.util.util import Util

class BaseGrid(object):
    def __init__(self, R):
        self.GRID_DIR = C.GRID_DIR
        self.wave = None
        self.flux = None
        self.Res = 5000
        self.R = R
        self.RR = C.dRR[R]
        self.Util = Util()
        self.step=10
        self.Ws = [7100, 8850]

    def load_grid(self, PATH=None):
        if PATH is None: PATH = os.path.join(self.GRID_DIR, f"bosz_{self.Res}_{self.RR}.h5")
        with h5py.File(PATH, "r") as f:
            self.wave0 = f["wave"][()]
            self.flux0 = f["flux"][()]
            self.para = f["para"][()]
            self.pdx = f["pdx"][()]
        print(self.wave0.shape, self.flux0.shape, self.para.shape)
        self.pdx0 = self.pdx - self.pdx[0]
        
    def prepro(self, wave, flux):
        wave, flux = self.Util.get_flux_in_Wrange(wave, flux, self.Ws)
        wave, flux = self.Util.resample(wave, flux, self.step)
        return wave, flux

    def get_pdx_idx(self, pdx_i):
        mask = True
        for ii, p in enumerate(pdx_i):
            mask = mask & (self.pdx0[:,ii] == p)
        idx = np.where(mask)[0][0]
        return idx


    def build_rbf(self, flux):
        print(f"Building RBF on flux shape {flux.shape}")
        self.rbf = RBFInterpolator(self.pdx0, flux, kernel='gaussian', epsilon=0.5)

    def prepare(self):
        wave, flux = self.prepro(self.wave0, self.flux0)
        self.wave = wave
        self.flux = flux
        self.build_rbf(flux)
        self.pca(flux)

    def pca(self, flux, top=10):
        normlog_flux = self.Util.normlog_flux(flux)
        _,s,v = np.linalg.svd(normlog_flux, full_matrices=False)
        self.eigv0 = v
        if top is not None: 
            v = v[:top]
            s=s[:top]
        print(s[:10].round(2))
        assert abs(np.mean(np.sum(v.dot(v.T), axis=0)) -1) < 1e-5
        assert abs(np.sum(v, axis=1).mean()) < 0.1
        self.eigv = v
        return v


    def init_para(self, para):
        return pd.DataFrame(data=para, columns=C.pshort)

    def load_para(self):
        dfpara = pd.read_csv(self.GRID_DIR + "para.csv")
        return dfpara

    def get_flux_in_Prange(self, R=None, para=None, fix_CA=False):
        if para is None: 
            dfpara = self.load_para()
        else:
            dfpara = self.init_para(para)
        if R is None: 
            R = self.R
            
        Fs, Ts, Ls,_,_ = C.dRs[R]
        if fix_CA:
            dfpara = dfpara[(dfpara["A"] == 0.0)]
            # dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["A"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        maskM = (dfpara["M"] >= Fs[0]) & (dfpara["M"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["G"] >= Ls[0]) & (dfpara["G"] <= Ls[1]) 
        mask = maskM & maskT & maskL
        self.dfpara = dfpara[mask]
        self.para = np.array(self.dfpara.values, dtype=np.float16)
        return self.dfpara.index

    def save_dataset(self, wave, flux, para, SAVE_PATH=None):
        if SAVE_PATH is None: SAVE_PATH = os.path.join(self.GRID_DIR, f"bosz_{self.Res}_{self.RR}.h5")
        with h5py.File(SAVE_PATH, "w") as f:
            f.create_dataset(f"flux", data=flux, shape=flux.shape)
            f.create_dataset(f"para", data=para, shape=para.shape)
            f.create_dataset(f"wave", data=wave, shape=wave.shape)  
