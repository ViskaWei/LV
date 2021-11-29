import numpy as np
import cupy as cp
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

from matplotlib import pyplot as plt
from lv.util.constants import Constants
from lv.util.util import Util


class BasePCA(object):
    def __init__(self, dataset_nm="dataset", grid=1, prepro=1, CUDA=1):
        self.DATADIR = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/{dataset_nm}/"
        self.prepro=prepro
        self.CUDA=CUDA
        # self.top= None
        self.c = Constants()
        self.dWs = self.c.dWs
        # self.dRs = c.dRs
        self.dR = self.c.dR
        self.Rnms = self.c.Rnms
        self.Cnms = self.c.Cnms
        # self.RRnms = c.RRnms
        self.nFlux = {}
        self.nPara = {}
        self.nErr = {}
        self.nSNR = {}
        self.nVs = {}
        self.pcFlux = {}
        self.npcFlux = {}
        self.nwave = None
        self.W = None
        self.Util = Util()
        self.grid=grid

    def run(self, W, Rs=None, N=1000, top=100, transform=0, save=0):
        self.prepare_data_W(W, Rs, N=N)
        self.prepare_svd(W, Rs, top=top, transform=transform)
        if save: 
            name = "" if N is None else f"_{N // 1000}k"
            self.save_W(W, name=name)


    def run_R(self, W, R, N=None, top=100, transform=0, name="", save=1):
        self.prepare_data_W_R(W, R,N=N)
        self.prepare_svd_R(R, top=top, transform=transform)
        if save: 
            self.collect_PC_W_R(W, R, name=name)
        else:
            self.plot_pcFlux_R(R)
        return self.nVs[R]

    def prepare_data_W(self, W, Rs=None, N=1000):
        if Rs is None: Rs = self.Rnms
        for R in tqdm(Rs):
            self.prepare_data_W_R(W, R, N=N)

    def dataloader_W_R(self, W="RML", R=None, N=None, mag=19):
        RR = self.dR[R]
        Ws = self.dWs[W]
        if N is not None:
            nn = N // 1000
            DATAPATH = self.DATADIR + f"{RR}/sample/{Ws[3]}_R{Ws[2]}_{nn}k_m{mag}.h5"
        else:
            DATAPATH = self.DATADIR + f"{RR}/grid/{Ws[3]}_R{Ws[2]}_m{mag}.h5"

        with h5py.File(DATAPATH, 'r') as f:
            wave = f['wave'][()]
            flux = f['flux'][()]
            pval = f['pval'][()]
        print(f"# {RR} flux: {flux.shape}, wave {W}: {wave.shape} ")
        return wave, flux, pval

    def prepro_fn(self, wave, flux):
        return wave, self.Util.lognorm_flux(flux)

    def prepare_data_W_R(self, W, R, N=10000):
        wave, flux, pval = self.dataloader_W_R(W, R, N=N)
        self.nwave = wave
        if self.prepro:
            wave, flux = self.prepro_fn(wave, flux)
        self.nFlux[R] = flux
        self.nPara[R] = pval
    
    def prepare_svd_R(self, R, top=100, transform=0):
        flux_R = self.nFlux[R]
        flux_Rc= cp.asarray(flux_R, dtype=cp.float32)
        Vs = self.get_eigv(flux_Rc, top=top)
        self.nVs[R] = cp.asnumpy(Vs)
        if transform:
            self.pcFlux[R] = cp.dot(flux_Rc, Vs.T)
            self.npcFlux[R] = cp.asnumpy(self.pcFlux[R])

    def prepare_svd(self, W, Rs=None, top=100, transform=0):
        if Rs is None: Rs = self.Rnms
        for R in Rs:
            self.prepare_svd_R(R, top=top, transform=transform)

    def get_eigv(self, X, top=5):
        if self.CUDA:
            _,_,v = self._svd(X)
        else:
            _,_,v = np.linalg.svd(X)
        return v[:top] 

    def _svd(self, X):
       return cp.linalg.svd(X, full_matrices=0)

        
    def save_W_R(self, W, R, PATH=None, name=""):
        Ws = self.dWs[W]
        if PATH is None: 
            PATH=f"/scratch/ceph/swei20/data/dnn/PC/logPC/{Ws[3]}_R{Ws[2]}{name}.h5"
        print(PATH)
        nV = self.nVs[R]
        with h5py.File(PATH, "w") as f:
            f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)

    def save_W(self, W, PATH=None, name=""):
        Ws = self.dWs[W]
        if PATH is None: 
            PATH=f"/scratch/ceph/swei20/data/dnn/PC/logPC/{Ws[3]}_R{Ws[2]}{name}.h5"
        print(PATH)
        with h5py.File(PATH, "w") as f:
            for R, nV in self.nVs.items():
                f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)

    def plot_pcFlux_R(self, R, data=None, idx0=0, idx1=1, pdx=1):
        if data is None:
            data = self.npcFlux[R]
        plt.figure(figsize=(5,4), facecolor='w')
        plt.scatter(data[:,idx0], data[:,idx1],c=self.nPara[R][:,pdx], s=1, cmap = self.Cnms[pdx], label=f"{self.c.Pnms[pdx]}")
        plt.xlabel(f"PC{idx0}")
        plt.ylabel(f"PC{idx1}")
        plt.legend()
        plt.colorbar()

    def plot_V_R(self, R,top=5, step=0.3, ax=None):
        self.plot_V(self.nVs[R], top=top, step=step, ax=ax)

    def plot_V(self, nv, top=5, step=0.3, ax=None):
        size = top // 5
        if ax is None: ax = plt.subplots(1, figsize=(16,5 * size),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(self.nwave, nv[i] + step*(i+1))
        self.get_wave_axis(self.nwave, ax=ax)
        
    def get_wave_axis(self, wave, ax=None, xgrid=True):
        ax.set_xlim(wave[0]-1, wave[-1]+2)
        ax.set_xticks(np.arange(int(wave[0]), np.ceil(wave[-1]), 200))
        ax.xaxis.grid(xgrid)

    def get_rf(self, data, lbl):
        rf = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=100, max_features=30)
        rf.fit(data, lbl)
        return rf.feature_importances_

    def barplot_rf(self, fi, sdx=None, top=15, log=0, color="k", ax=None):
        if sdx is None: sdx = fi.argsort()[::-1][:top]
        if ax is None: ax =plt.subplots(1, figsize=(16,1), facecolor="w")[1]
        ax.bar([f"{sdx[i]}" for i in range(len(sdx))],  fi[sdx], log=log, color=color)

    # def plot_nV


    def pcloader_W(self, W=None, Rs=None, top=100, name=""):
        if Rs is None: Rs = self.Rnms
        Ws = self.dWs[W]
        PC_PATH = f"/scratch/ceph/swei20/data/dnn/PC/logPC/{Ws[3]}_R{Ws[2]}{name}.h5"
        dPC = {}
        with h5py.File(PC_PATH, 'r') as f:
            for R in Rs:
                PC = f[f'PC_{R}'][()]
                dPC[R] = PC[:top]
        nPixel = PC.shape[1]        
        return dPC, nPixel



    # def resample(self, wave, flux, err=None, step=20, verbose=1):
    #     if err is None:
    #         return u.resample(wave, flux, step=step, verbose=verbose)
    #     else:
    #         return u.resample_ns(wave, flux, err, step=step, verbose=verbose)
    
    # def resampleFlux(self, flux, L, step):
    #     return u.resampleFlux(flux, L, step=step)