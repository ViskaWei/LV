#---------------------------------------------------------
# View the 2D/3D parameter grid for the BOSZ models
#---------------------------------------------------------
import numpy as np
import scipy as sp
import pandas as pd
import os
import sys
import collections
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
from scipy import stats
from lv.pca.basePCA import BasePCA
from lv.constants import Constants as c
from lv.util import Util as u

class Box():
    def __init__(self):
        self.para = None


    def init_para(self):
        NORM_PATH = "/scratch/ceph/szalay/swei20/AE/norm_flux.h5"
        with h5py.File(NORM_PATH, 'r') as f:
            para = f['para'][()]
        MH    = para[:,0]
        Teff  = para[:,1]
        logG  = para[:,2]
        CM    = para[:,3]
        ALPHA = para[:,4]
        uM = np.unique(MH)
        uT = np.unique(Teff)
        uG = np.unique(logG)
        uC = np.unique(CM)
        uA = np.unique(ALPHA)


    def get_rbf_cmd(self, R=None, boszR=5000):
        print(R, sep="/n/n")
        pp = c.dRs[R][:3]
        base = "./scripts/build_rbf.sh grid bosz --config ./configs/import/stellar/bosz/rbf/"
        ins = f" --in /scratch/ceph/dobos/data/pfsspec/import/stellar/grid/bosz_{boszR}"
        out =  f" --out /scratch/ceph/swei20/data/pfsspec/import/stellar/rbf/bosz_{boszR}_{c.dRR[R]}/"
        param = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} "
        cmd = base + ins+ out + param
        print(cmd)
    
    def step2_RBF(self, mkdir=1):
        if mkdir: 
            for R in c.Rnms:
                os.mkdir(f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{c.dRR[R]}/")  
        for R in c.Rnms:
            self.get_rbf_cmd(R)

    def get_noise_cmd(self, R, W="RedM", N=10000, boszR=5000, pixelR=5000):
        assert N >= 1000
        print(R, sep="/n/n")
        pp = c.dRs[R]
        w  = c.dWw[W][0]
        nn = N // 1000
        base = "./scripts/prepare.sh model bosz-rbf pfs --config ./configs/infer/pfs/bosz/nowave/prepare/train.json"
        arm  = f"  ./configs/infer/pfs/bosz/nowave/inst_pfs_{w}.json"
        size = f" --chunk-size 1000 --sample-count {N}"
        inD  = f" --in /scratch/ceph/swei20/data/pfsspec/import/stellar/rbf/bosz_{boszR}_{c.dRR[R]}/rbf"
        outD = f" --out /scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{c.dRR[R]}/bosz_{pixelR}_{W}_{nn}k"
        para = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} --C_M {pp[3][0]} {pp[3][1]} --O_M {pp[4][0]} {pp[4][1]}"
        mag  = f" --mag-filter /scratch/ceph/dobos/data/pfsspec/subaru/hsc/hsc_i.dat --mag 16 18.5"
        cmd = base + arm + size + inD + outD + para + mag
        print(cmd)

    def step3_noise(self, N=1000):
        for R in c.Rnms:
            self.get_noise_cmd(R, N=N)

    def load_dataset(self, R, W="RedM", N=1000, pixelR=5000):
        RR = c.dRR[R]
        nn = N // 1000
        DATA_PATH=f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/bosz_{pixelR}_{W}_{nn}k/dataset.h5"

        with h5py.File(DATA_PATH, "r") as f:
            wave = f["wave"][()]
            flux = f["flux"][()]
            mask = f["mask"][()]
            error = f["error"][()]
        assert (np.sum(mask) == 0)
        dfparams = pd.read_hdf(DATA_PATH, "params")    
        para = dfparams[['Fe_H','T_eff','log_g','C_M','O_M']].values
        # snr = dfparams[['redshift','mag','snr']].values
        snr = dfparams['snr'].values
        return wave, flux, error, para, snr

    def save_dataset(self, wave, flux, error, para, snr, SAVE_PATH):
        with h5py.File(SAVE_PATH, "w") as f:
            f.create_dataset(f"logflux", data=flux, shape=flux.shape)
            f.create_dataset(f"pval", data=para, shape=para.shape)
            f.create_dataset(f"wave", data=wave, shape=wave.shape)  
            f.create_dataset(f"error", data=error, shape=error.shape)          
            f.create_dataset(f"snr", data=snr, shape=snr.shape)

    def convert(self, R, W="RedM", N=1000, boszR=5000, step=20):
        wave, flux, error, para, snr = self.load_dataset(R, W=W, N=N)  
        RR = c.dRR[R]
        nn = N // 1000
        SAVE_PATH = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/{W}_R{boszR}_{nn}k.h5"
        self.save_dataset(wave, flux, error, para, snr, SAVE_PATH)    
        waveL, fluxL = u.resample(wave, flux, step=step)  
        print(flux.shape, fluxL.shape, error.shape, para.shape)
        w = c.dWw[W][1]
        ws = c.dWs[w]
        SAVE_PATHL = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/{ws[3]}_R{ws[2]}_{nn}k.h5"
        self.save_dataset(waveL, fluxL, error, para, snr, SAVE_PATHL)
    
    def resample(self,wave, flux, step=20):
        return u.resample(wave, flux, step=step)

    def resampleFlux_i(self, flux, step=20):
        return u.resampleFlux_i(flux, step=step)
    
    def step4_downSample(self, N=1000, step=20):
        for R in c.Rnms:
            self.convert(R, N=N, step=step)

    def step5_PCA(self, W="RML", N=1000, top=100, name=""):
        p = BasePCA()
        p.run(W, N=N, top=top, transform=0, save=1, name=name)
        

