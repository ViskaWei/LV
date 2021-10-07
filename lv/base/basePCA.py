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
        self.Ws = c.Ws
        self.Rs = c.Rs

        self.Nms = {"M": "M31 Giant", "W": "MW Warm", "C": "MW Cool", "B": "BHB", "R": "RHB", "G":"DwarfG Giant"}
        self.Flux = {}
        self.nFlux = {}
        self.nPara = {}
        self.pcFlux = {}
        self.npcFlux = {}
        self.Size = {}
        self.Vs = {}
        self.nVs = {}
        self.Ms = {}
        self.Ns = {}
        self.Mvs = {}
        self.Nvs = {}
        self.wave = None
        self.nwave = None
        self.mean = None
        self.size = None
        self.center = False
        self.prod = None 
        self.nf = None
        self.nw = None
        self.mask = None
        self.nmask = None
        self.Fs = {}
        self.cmap="YlGnBu"
        self.color = c.Cs_
        self.pnames= c.Ps
        self.ps =  [["p0","p1", "p2", "p3", "p4"],["p5","p6", "p7", "p8", "p9"],["p10","p11", "p12", "p13", "p14"],["p15","p16", "p17", "p18", "p19"]]
        self.name = None
        self.lick = None

####################################### Flux #####################################
    # def load_data(W):

    def prepare_data(self, flux, wave, para0, W=None, fix_CO=False):
        # flux = np.clip(-flux, 0.0, None)
        if W is not None: self.W = self.Ws[W]
        self.nwave = wave        

        for p, pvals in self.Rs.items():
            index, para = self.get_flux_in_Prange(para0, pvals, fix_CO=fix_CO)
             # flux, wave = self.get_flux_in_Wrange(flux, wave)
            flux_p = flux[index]
            self.nFlux[p] = -flux_p
            self.nPara[p] = para

            self.Size[p] = flux_p.shape[0]
            print(f"# {p} flux: {self.Size[p]}, wave {W}: {wave.shape} ")

    def stack_data(self,save=0, PATH=None):
        fluxs = []
        lbls = []
        paras = []
        for ii, (key, flux) in enumerate(self.nFlux.items()):
            if key != "G":
                para = self.nPara[key]
                n = len(flux)
                fluxs.append(flux)
                paras.append(para)
                lbls.append(np.zeros(n) + ii)
        fluxs = np.vstack(fluxs)
        paras = np.vstack(paras)
        lbls = np.hstack(lbls)
        print(fluxs.shape, paras.shape, lbls.shape)
        if save:
            if PATH is None: PATH = f"/scratch/ceph/swei20/data/dnn/ALL/norm_flux.h5" 
            ww = self.W[3][:1]
            with h5py.File(PATH, "w") as f:
                f.create_dataset(f"flux{ww}", data=fluxs, shape=fluxs.shape)
                f.create_dataset(f"lbl{ww}", data=lbls, shape=lbls.shape)
                f.create_dataset(f"para{ww}", data=paras, shape=paras.shape)
        return fluxs, paras, lbls

    def prepare_svd(self, top=50, save=0, PATH=None):
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
        if PATH is None: PATH = f"/scratch/ceph/swei20/data/dnn/pc/bosz_{self.W[3]}_R{self.W[2]}.h5"
        print(PATH)
        with h5py.File(PATH, "w") as f:
            for p, nV in self.nVs.items():
                para = self.nPara[p]
                f.create_dataset(f"PC_{p}", data=nV, shape=nV.shape)
                f.create_dataset(f"para_{p}", data=para, shape=para.shape)


    def prepare_rf(self, cut=12):
        for Rgn in self.Rs.keys():
            self.get_rfdx(Rgn, cut=cut)


    def save_PCA(self, PATH):
        with h5py.File(PATH, "w") as f:
            for p, nV in self.nVs.items():
                f.create_dataset(f"pc{p}", data=nV, shape=nV.shape)
                f.create_dataset(f"pcFlux{p}", data=self.npcFlux[p], shape=self.npcFlux[p].shape)

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

    def get_wave_axis(self, wave= None, ax=None, xgrid=True):
        if wave is None: 
            ax.set_xlim(self.nwave[0]-1, self.nwave[-1]+2)
            ax.set_xticks(np.arange(self.W[0], self.W[1], 200))
            
        else:
            ax.set_xlim(wave[0]-1, wave[-1]+2)
            ax.set_xticks(np.arange(int(wave[0]), np.ceil(wave[-1]), 200))
        ax.xaxis.grid(xgrid)
####################################### SVD  #######################################

    def _svd(self, X):
        return cp.linalg.svd(X, full_matrices=0)

    def get_eigv(self, X, top=5):
        _,_,v = self._svd(X)
        return v[:top] 

    def plot_Vs(self, top=5, step=0.3):
        wave = self.nwave
        f, axs = plt.subplots(6,1, figsize=(16,18),facecolor="w")
        for i, (p, nv) in enumerate(self.nVs.items()):
            ax = axs[i]
            self.plot_Vs_p(p, ax=ax, top=top, step=step)
            
    def plot_Vs_p(self, p, vs=None, top=5, step=0.3, ax=None):
        wave = self.nwave
        if vs is None: vs = self.nVs
        nv = self.nVs[p]
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(wave, nv[i] + step*(i+1))
        ax.set_ylabel(f"{self.Nms[p]}")
        self.get_wave_axis(wave=wave, ax=ax)

    def plot_V(self, nv, top=5, step=0.3, ax=None):
        wave = self.nwave        
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(wave, nv[i] + step*(i+1))
        self.get_wave_axis(wave=wave, ax=ax)

####################################### Mask #######################################
    def barplot_rf(self, fi, sdx=None, top=15, log=0, color="k", ax=None):
        if sdx is None: sdx = fi.argsort()[::-1][:top]
        if ax is None: ax =plt.subplots(1, figsize=(16,1), facecolor="w")[1]
        ax.bar([f"{sdx[i]}" for i in range(len(sdx))],  fi[sdx], log=log, color=color)
        return list(sdx)

    def get_rf(self, data, lbl):
        rf = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=100, max_features=30)
        rf.fit(data, lbl)
        return rf.feature_importances_

    def get_rfdx(self,Rgn, cut=20):
        pcFlux=self.npcFlux[Rgn]
        lbl = self.nPara[Rgn]
        print(pcFlux.shape, lbl.shape)
        cut0 = np.max((12, cut//3 +1))
        sdx=[]
        for idx in [0,1,2]:
            fi = self.get_rf(pcFlux, lbl[:,idx])
            sdx.append(self.barplot_rf(fi, top=cut0))
            plt.annotate(f"{self.Nms[Rgn]}\n{self.pnames[idx]}", (0.9,0.2), xycoords="axes fraction", fontsize=20)
        sdxx = self.get_sdxx_from_sdx(sdx, top=8, cut=cut)
        self.Fs[Rgn] = sdxx

    def get_sdxx_from_sdx(self, sdx, top=5, cut=12):
        sdxx = set(sdx[1][:top] + sdx[2][:top])
        i=0
        j=0
        while (len(sdxx) < cut) & (j < len(sdx[1])-5):
            try: 
                sdxx.add(sdx[0][i])
            except:
                sdxx.add(sdx[1][top+j])
                j+1
            i+=1 
        return list(sdxx)


    def plot_v(self, vs, idx, nidx=None, c=None, ax=None):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,5))[1]
        vs = cp.asnumpy(vs)
        v = vs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)
        
    def plot_nv(self, nvs, idx, nidx=None, c=None, ax=None):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,5))[1]
        v = nvs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)

    def set_unique_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())















