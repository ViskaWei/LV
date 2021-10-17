import numpy as np
import cupy as cp
import pandas as pd
import h5py
from tqdm import tqdm

from matplotlib import pyplot as plt
from lv.constants import Constants as c
from lv.util import Util as u


class BasePCA(object):
    def __init__(self):
        # self.top= None
        self.dWs = c.dWs
        # self.dRs = c.dRs
        self.dR = c.dR
        self.Rnms = c.Rnms
        self.Cnms = c.Cnms
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

    def run(self, W, N=1000, top=100, transform=0,name="", save=0):
        self.prepare_data_W(W, N=N)
        self.prepare_svd(W, top=top, transform=transform)
        if save: self.collect_PC(W, name=name)


    def run_R(self, W, R, N=None, top=100, transform=0, name="", save=1):
        self.prepare_data_W_R(W, R,N=N)
        self.prepare_svd_R(R, top=top, transform=transform)
        if save: 
            self.collect_PC_W_R(W, R, name=name)
        else:
            self.plot_pcFlux_R(R)
        return self.nVs[R]

    def prepare_data_W(self, W, N=1000):
        for R in tqdm(self.Rnms):
            self.prepare_dataset_W_R(W, R, N=N)

    def load_RBF_W_R(self, W="RML", R=None, N=1000, pdx=None):
        nn= N // 1000
        RR = self.dR[R]
        Ws = self.dWs[W]
        DATA_PATH = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/{Ws[3]}_R{Ws[2]}_{nn}k.h5"
        with h5py.File(DATA_PATH, 'r') as f:
            wave = f['wave'][()]
            flux = f['logflux'][()]
            pval = f['pval'][()]
            # error = f['error'][()]
            # snr =   f['snr'][()]
        # if pdx is not None: pval = pval[:, pdx]
        print(f"# {RR} flux: {flux.shape}, wave {W}: {wave.shape} ")
        return wave, flux, pval

    def prepare_data_W_R(self, W, R, N=10000):
        wave, flux, pval = self.load_RBF_W_R(W, R, N=N)
        self.nwave = wave
        self.nFlux[R] = flux
        self.nPara[R] = pval

    def prepare_data_W(self, W, N):
        for R in self.Rnms:
            self.prepare_data_W_R(W, R, N=N)
    
    def prepare_svd_R(self, R, top=100, transform=0):
        flux_R = self.nFlux[R]
        flux_Rc= cp.asarray(flux_R, dtype=cp.float32)
        Vs = self.get_eigv(flux_Rc, top=top)
        self.nVs[R] = cp.asnumpy(Vs)
        if transform:
            self.pcFlux[R] = cp.dot(flux_Rc, Vs.T)
            self.npcFlux[R] = cp.asnumpy(self.pcFlux[R])

    def prepare_svd(self, W, top=100, transform=0):
        for R in self.Rnms:
            self.prepare_svd_R(R, top=top, transform=transform)

    def get_eigv(self, X, top=5):
        _,_,v = self._svd(X)
        return v[:top] 

    def _svd(self, X):
       return cp.linalg.svd(X, full_matrices=0)

        
    def collect_PC_W_R(self, W, R, PATH=None, name=""):
        Ws = self.dWs[W]
        if PATH is None: 
            PATH=f"/scratch/ceph/swei20/data/dnn/PC/logPC/{Ws[3]}_R{Ws[2]}{name}.h5"
        print(PATH)
        nV = self.nVs[R]
        with h5py.File(PATH, "w") as f:
            f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)

    def collect_PC(self, W, PATH=None, name=""):
        Ws = self.dWs[W]
        if PATH is None: 
            PATH=f"/scratch/ceph/swei20/data/dnn/PC/logPC/{Ws[3]}_R{Ws[2]}{name}.h5"
        print(PATH)
        with h5py.File(PATH, "w") as f:
            for R, nV in self.nVs.items():
                f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)

    def plot_pcFlux_R(self, R, idx0=0, idx1=1, pdx=1):
        plt.figure(figsize=(5,4), facecolor='w')
        plt.scatter(self.npcFlux[R][:,idx0], self.npcFlux[R][:,idx1],c=self.nPara[R][:,pdx], s=1, cmap = self.Cnms[pdx])
        plt.xlabel(f"PC{idx0}")
        plt.ylabel(f"PC{idx1}")
        plt.colorbar()

    # def plot_nV



    # def resample(self, wave, flux, err=None, step=20, verbose=1):
    #     if err is None:
    #         return u.resample(wave, flux, step=step, verbose=verbose)
    #     else:
    #         return u.resample_ns(wave, flux, err, step=step, verbose=verbose)
    
    # def resampleFlux(self, flux, L, step):
    #     return u.resampleFlux(flux, L, step=step)