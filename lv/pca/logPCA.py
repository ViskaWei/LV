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
from lv.util import Util as u

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


    def load_dataset(self, DATA_PATH):
        with h5py.File(DATA_PATH, "r") as f:
            wave = f["wave"][()]
            flux = f["flux"][()]
            mask = f["mask"][()]
            error = f["error"][()]
        assert (np.sum(mask) == 0)
        dfparams = pd.read_hdf(DATA_PATH, "params")    
        dfpara = dfparams[['Fe_H','T_eff','log_g','C_M','O_M']]
        dfsnr = dfparams[['redshift','mag','snr']]
        return wave, flux, error, dfpara, dfsnr

    def resample(self, wave, flux, step=10, verbose=1):
        return u.resample(wave, flux, step=step, verbose=verbose)
    




    def get_PC(self, wave, flux, para, top=50, save=1, Ws=["RML"]):
        if wave is None:
            DATA_PATH =  f"/scratch/ceph/swei20/data/pfsspec/import/stellar/grid/bosz_R1000/logflux.h5"
            with h5py.File(DATA_PATH, 'r') as f:
                flux = f['flux'][()]
                para = f['para'][()]
                wave = f['wave'][()]
        for W in Ws:
            wave = self.get_PC_W(W, wave, flux, para, top=top, save=save)
        return wave

    def get_PC_W(self, W, wave0, flux0, para0, top=50, save=0, PATH=None):
        W = self.dWs[W]
        wave, flux = self.get_flux_in_Wrange(wave0, flux0, W)
        for R, pvals in self.dRs.items():
            index, _ = self.get_flux_in_Prange(para0, pvals)
             # flux, wave = self.get_flux_in_Wrange(flux, wave)
            flux_R = flux[index]
            flux_Rc= cp.asarray(flux_R, dtype=cp.float32)
            Vs = self.get_eigv(flux_Rc, top=top)
            self.nVs[R] = cp.asnumpy(Vs)
        if save:
            self.collect_PC(W, PATH)
        return wave

    def get_eigv(self, X, top=5):
        _,_,v = self._svd(X)
        return v[:top] 

    def _svd(self, X):
        return cp.linalg.svd(X, full_matrices=0)

    def get_flux_in_Prange(self, para0, pval):
        Fs, Ts, Ls, _,_ = pval
        dfpara = self.init_para(para0)
        maskF = (dfpara["F"] >= Fs[0]) & (dfpara["F"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["L"] >= Ls[0]) & (dfpara["L"] <= Ls[1]) 
        mask = maskF & maskT & maskL
        dfpara = dfpara[mask]
        para = np.array(dfpara.values, dtype=np.float16)
        return dfpara.index, para

    def get_flux_in_Wrange(self, wave, flux, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return wave[start:end], flux[:, start:end]

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
    
    def collect_PC(self, W, PATH=None):
        if PATH is None: PATH = f"/scratch/ceph/swei20/data/dnn/PC/logPC/bosz_{W[3]}_R{W[2]}.h5"
        print(PATH)
        with h5py.File(PATH, "w") as f:
            for R, nV in self.nVs.items():
                f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)

    # def plot_nvs(self, nvs, c="k", ax=None, fineW=0):
    #     n = len(idxs)
    #     f, axs = plt.subplots(n,1, figsize=(16,2*n))
    #     for i in range(n):
    #         ax = axs[i]
    #         nv = nvs[i]
    #         ax.plot(self.nwave, nv, c=c)
    #         if not fineW: 
    #             self.get_wave_axis(ax=ax)
    #         else:
    #             ax.xaxis.grid(1)
    #         ax.legend(loc=1)

    def plot_V(self, wave, nv, top=5, step=0.3, ax=None):
        size = top // 5
        if ax is None: ax = plt.subplots(1, figsize=(16,3 * size),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(wave, nv[i] + step*(i+1))
        self.get_wave_axis(ax=ax)