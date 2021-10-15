import numpy as np
import cupy as cp
import pandas as pd
import h5py
from tqdm import tqdm

from lv.constants import Constants as c
from lv.util import Util as u


class BasePCA(object):
    def __init__(self, W=None, top=100):
        self.top=top
        self.dWs = c.dWs
        self.dRs = c.dRs
        self.dRR = c.dRR
        self.Rnms = c.Rnms
        self.RRnms = c.RRnms
        self.nFlux = {}
        self.nPara = {}
        self.nErr = {}
        self.nSNR = {}
        self.nVs = {}
        self.pcFlux = {}
        self.npcFlux = {}
        self.nwave = None
        self.W = self.dWs[W] if W is not None else None

    def prepare_dataset_W(self, W=None, N=1000, step=None):
        for R in tqdm(self.Rnms):
            self.prepare_dataset_R_W(R, W=W, N=N, step=step)
        

    def prepare_dataset_R_W(self, R, W=None, N=10000, step=None):
        if W is None: W=self.W
        nn= N // 1000
        RR = self.dRR[R]
        DATA_PATH = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/bosz_5000_{W[3]}_{nn}k/dataset.h5"  
        wave, flux, error, para, dfsnr = self.load_dataset(DATA_PATH)
        if step is not None:
            wave, flux = self.resample(wave, flux, step=step, verbose=1)
        if self.nwave is None: self.nwave = wave
        self.nFlux[R] = flux
        self.nPara[R] = para
        self.nErr[R] = error
        self.nSNR[R] = dfsnr['snr'].values
        print(f"# {RR} flux: {flux.shape}, wave {W}: {wave.shape} ")


    def load_dataset(self, DATA_PATH):
        with h5py.File(DATA_PATH, "r") as f:
            wave = f["wave"][()]
            flux = f["flux"][()]
            mask = f["mask"][()]
            error = f["error"][()]
        assert (np.sum(mask) == 0)
        dfparams = pd.read_hdf(DATA_PATH, "params")    
        para = dfparams[['Fe_H','T_eff','log_g','C_M','O_M']].values
        dfsnr = dfparams[['redshift','mag','snr']]
        return wave, flux, error, para, dfsnr

    def resample(self, wave, flux, err=None, step=20, verbose=1):
        if err is None:
            return u.resample(wave, flux, step=step, verbose=verbose)
        else:
            wL, fL = u.resample(wave, flux, step=step, verbose=verbose)
            eL = u.resampleFlux(flux + err, len(wL), step=step)
            return wL, fL, eL
    
    def resampleFlux(self, flux, L, step):
        return u.resampleFlux(flux, L, step=step)

    def prepare_svd_R(self, R, top=100, transform=0):
        flux_R = self.nFlux[R]
        flux_Rc= cp.asarray(flux_R, dtype=cp.float32)
        Vs = self.get_eigv(flux_Rc, top=top)
        self.nVs[R] = cp.asnumpy(Vs)
        if transform:
            self.pcFlux[R] = cp.dot(flux_Rc, Vs.T)
            self.npcFlux[R] = cp.asnumpy(self.pcFlux[R])

    def prepare_svd(self, W=None, top=100, name="", transform=0):
        W = self.W if W is None else self.dWs[W]
        for R in self.Rnms:
            self.prepare_svd_R(R, top=top, transform=transform)
        self.collect_PC(W=W, name=name)

    def get_eigv(self, X, top=5):
        _,_,v = self._svd(X)
        return v[:top] 

    def _svd(self, X):
       return cp.linalg.svd(X, full_matrices=0)

        
    def get_PC_R(self, R, W=None, N=None, step=20, transform=0):
        self.prepare_dataset_R_W(R, W=W, N=N, step=step)
        self.prepare_svd_R(R, top=self.top, transform=transform)
        return self.nVs[R]

    def get_PC(self, N=1000, step=20):
        for R in self.Rnms:
            _=self.get_PC_R(R, N=N, step=step)

    def collect_PC_R(self, R, W=None, PATH=None, name=None):
        if W is None: W=self.W 
        if PATH is None: 
            PATH=f"/scratch/ceph/swei20/data/dnn/PC/logPC/bosz_{W[3]}_R{W[2]}{name}.h5"
        print(PATH)
        nV = self.nVs[R]
        with h5py.File(PATH, "w") as f:
            f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)

    def collect_PC(self, W=None, PATH=None, name=None):
        if PATH is None: 
            PATH=f"/scratch/ceph/swei20/data/dnn/PC/logPC/bosz_{W[3]}_R{W[2]}{name}.h5"
        print(PATH)
        with h5py.File(PATH, "w") as f:
            for R, nV in self.nVs.items():
                f.create_dataset(f"PC_{R}", data=nV, shape=nV.shape)

