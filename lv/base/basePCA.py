import time
# import fbpca
import logging
import numpy as np
import cupy as cp
import pandas as pd
# from scipy.sparse.linalg import svds
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from tqdm import tqdm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from lv.constants import Constants as c


class PCA(object):
    def __init__(self):
        ################################ Flux Wave ###############################
        self.dWs = c.dWs
        self.dRs = c.dRs

        self.Nms = {"M": "M31 Giant", "W": "MW Warm", "C": "MW Cool", "B": "BHB", "R": "RHB", "G":"DwarfG Giant"}
        self.Flux = {}
        self.nFlux = {}
        self.nPara = {}
        self.pcFlux = {}
        self.npcFlux = {}
        self.Size = {}
        self.Vs = {}
        self.nVs = {}

        self.wave = None
        self.nwave = None
        self.nf = None
        self.nw = None
        self.Pnms= c.Pnms

####################################### Flux #####################################
    # def load_data(W):

    def prepare_data(self, flux, wave, para0, W=None, fix_CO=False):
        # flux = np.clip(-flux, 0.0, None)
        if W is not None: self.W = self.dWs[W]
        self.nwave = wave        

        for R, pvals in self.dRs.items():
            index, para = self.get_flux_in_Prange(para0, pvals, fix_CO=fix_CO)
             # flux, wave = self.get_flux_in_Wrange(flux, wave)
            flux_R = flux[index]
            self.nFlux[R] = flux_R
            self.nPara[R] = para

            self.Size[R] = flux_R.shape[0]
            print(f"# {R} flux: {self.Size[R]}, wave {W}: {wave.shape} ")



    def get_PC_W(self, W, wave, flux, para0, top=50, save=0, PATH=None):
        W = self.dWs[W]
        for R, pvals in self.dRs.items():
            index, para = self.get_flux_in_Prange(para0, pvals, fix_CO=0)
             # flux, wave = self.get_flux_in_Wrange(flux, wave)
            flux_R = flux[index]
        for R, flux_R in self.nFlux.items():
            flux_Rc= cp.asarray(flux_R, dtype=cp.float32)
            Vs = self.get_eigv(flux_Rc, top=top)
            self.nVs[p] = cp.asnumpy(Vs)
        if save:
            self.collect_PC(PATH)

    def get_flux_in_Prange(self, para0, p, fix_CO=True):
        Fs, Ts, Ls, _,_ = p
        dfpara = self.init_para(para0)
        if fix_CO:
            dfpara = dfpara[(dfpara["O"] == 0.0)]
            # dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["O"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        maskF = (dfpara["F"] >= Fs[0]) & (dfpara["F"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["L"] >= Ls[0]) & (dfpara["L"] <= Ls[1]) 
        mask = maskF & maskT & maskL
        dfpara = dfpara[mask]
        para = np.array(dfpara.values, dtype=np.float16)
        return dfpara.index, para

    def get_flux_in_Wrange(self, flux, wave):
        Ws = self.W
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]

    def init_para(self, para):
        return pd.DataFrame(data=para, columns=["F","T","L","C","O"])
        
    def prepare_svd_flux(self, top=50, save=0, PATH=None):
        #gpu only
        for p, flux_p in self.nFlux.items():
            self.Flux[p] = cp.asarray(flux_p, dtype=cp.float32)
        self.wave = cp.asarray(self.nwave, dtype=cp.float32)

        for p, flux_p in tqdm(self.Flux.items()):
            Vs = self.get_eigv(flux_p, top=top)
            self.Vs[p] = Vs
            self.pcFlux[p] = self.Flux[p].dot(Vs.T)
            self.npcFlux[p] = cp.asnumpy(self.pcFlux[p])
            self.nVs[p] = cp.asnumpy(Vs)
        if save:
            self.collect_PC(PATH)
    
    def collect_PC(self, PATH=None):
        if PATH is None: PATH = f"/scratch/ceph/swei20/data/dnn/PC/bosz_{self.W[3]}_R{self.W[2]}.h5"
        print(PATH)
        with h5py.File(PATH, "w") as f:
            for R, nV in self.nVs.items():
                f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)
