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
from lv.constants import Constants
from lv.util import Util

class Box():
    def __init__(self, volta=0):
        self.para = None
        self.slurm = ""
        self.c = Constants()
        self.Util = Util()

    def get_slurm(self, volta=0, srun=0, sbatch=0, mem=256):
        slurm=""
        if volta: 
            slurm = " -p v100"
        else:
            slurm = f" -p elephant --mem {mem}g"
        if sbatch:
            slurm = "sbatch" + slurm
        elif srun:
            slurm = "srun" + slurm
        self.slurm = slurm

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
        pp = self.c.dRs[R][:3]
        base = f"./scripts/build_rbf.sh {self.slurm} grid bosz --config ./configs/import/stellar/bosz/rbf/"
        ins = f" --in /scratch/ceph/dobos/data/pfsspec/import/stellar/grid/bosz_{boszR}"
        out =  f" --out /scratch/ceph/swei20/data/pfsspec/import/stellar/rbf/bosz_{boszR}_{self.c.dRR[R]}/"
        param = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} "
        cmd = base + ins+ out + param
        print(cmd)
        self.boszR=boszR
    
    def step2_RBF(self, mkdir=1, boszR=5000):
        if mkdir: 
            for R in self.c.Rnms:
                os.mkdir(f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{self.c.dRR[R]}/")  
        for R in self.c.Rnms:
            self.get_rbf_cmd(R, boszR=boszR)

    

    def get_pca_cmd(self, R, W="RedM", pixelR=5000, mag_lim=19, dataset_name=None):
        print(R, sep="/n/n")
        pp = self.c.dRs[R]
        w  = self.c.dWw[W][0]
        base = f"./scripts/prepare.sh {self.slurm} model bosz pfs --config ./configs/infer/pfs/bosz/nowave/prepare/train.json"
        arm  = f"  ./configs/infer/pfs/bosz/nowave/inst_pfs_{w}.json"
        size = f" --chunk-size 1 "
        inD  = f" --in /scratch/ceph/dobos/data/pfsspec/import/stellar/grid/bosz_50000"
        
        if self.dataset_name is None: self.dataset_name = f"bosz_R{pixelR}_{W}_m{mag_lim}"
        outD = f" --out /scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{self.c.dRR[R]}/grid/{self.dataset_name}"
        para = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} --C_M {pp[3][0]} {pp[3][1]} --O_M {pp[4][0]} {pp[4][1]}"
        norm = f" --norm none"
        mag  = f" --mag-filter /scratch/ceph/dobos/data/pfsspec/subaru/hsc/hsc_i.dat --mag {mag_lim} "
        grid = f" --sample-mode grid"

        cmd = base + arm + size + inD + outD + para + norm + mag + grid
        print(cmd)

    def step3_BuildPCA(self, R=None, W="RedM", pixelR=5000, mag_lim=19):
        self.mag_lim=mag_lim
        self.dataset_name = f"bosz_R{pixelR}_{W}_m{self.mag_lim}"
        self.pixelR=pixelR
        RList= self.c.Rnms if R is None else [R]
        for R in RList:
            self.get_pca_cmd(R, W=W, pixelR=pixelR, mag_lim=mag_lim, dataset_name=self.dataset_name)


    def get_noise_cmd(self, R, W="RedM", N=10000, pixelR=5000, dmag=1):
        assert N >= 1000
        print(R, sep="/n/n")
        pp = self.c.dRs[R]
        w  = self.c.dWw[W][0]
        nn = N // 1000
        base = f"./scripts/prepare.sh {self.slurm} model bosz-rbf pfs --config ./configs/infer/pfs/bosz/nowave/prepare/train.json"
        arm  = f"  ./configs/infer/pfs/bosz/nowave/inst_pfs_{w}.json"
        size = f" --chunk-size 1000 --sample-count {N}"
        inD  = f" --in /scratch/ceph/swei20/data/pfsspec/import/stellar/rbf/bosz_{self.boszR}_{self.c.dRR[R]}/rbf"
        self.dataset_name = f"rbf_R{pixelR}_{W}_{nn}k_m{self.mag_lim}"
        outD = f" --out /scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{self.c.dRR[R]}/{self.dataset_name}"
        para = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} --C_M {pp[3][0]} {pp[3][1]} --O_M {pp[4][0]} {pp[4][1]}"
        mag  = f" --mag-filter /scratch/ceph/dobos/data/pfsspec/subaru/hsc/hsc_i.dat --mag {self.mag_lim-dmag} {self.mag_lim+dmag}"
        norm = f" --norm none"
        
        cmd = base + arm + size + inD + outD + para + mag + norm
        print(cmd)
        self.pixelR=pixelR

    def step4_noise(self, N=1000):
        for R in self.c.Rnms:
            self.get_noise_cmd(R, N=N)

    def load_dataset(self, R, W="RedM", N=None, DATA_PATH=None, grid=False):
        RR = self.c.dRR[R]
        DATA_DIR =f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/"
        if grid: 
            DATA_DIR += "grid/"
            name = f"bosz_R{self.pixelR}_{W}_m{self.mag_lim}/"
        else:
            DATA_DIR += "rbf/"
            name = f"R{self.pixelR}_{W}_{N//1000}k_m{self.mag_lim}/"
        if DATA_PATH is None: 
            DATA_PATH = DATA_DIR + name + "dataset.h5"
        print(DATA_PATH)
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


    
    def convert(self, R, W="RedM", N=None, step=20, grid=0):
        wave, flux, error, para, snr = self.load_dataset(R, W=W, N=N, grid=grid)  
        RR = self.c.dRR[R]
        SAVE_DIR = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/"
        w = self.c.dWw[W][1]
        ws = self.c.dWs[w]
        if grid: 
            SAVE_DIR += f"grid/"
            name = f"{W}_R{self.pixelR}_m{self.mag_lim}.h5"
            nameL = f"{ws[3]}_R{ws[2]}_m{self.mag_lim}.h5"
        else:
            SAVE_DIR += f"rbf/"
            name = f"{W}_R{self.pixelR}_{N//1000}k_m{self.mag_lim}.h5"
            nameL = f"{ws[3]}_R{ws[2]}_{N//1000}k_m{self.mag_lim}.h5"
        if not os.path.isdir(SAVE_DIR):
            os.mkdir(SAVE_DIR)

        SAVE_PATH = SAVE_DIR + name
        self.save_dataset(wave, flux, error, para, snr, SAVE_PATH)    
        
        waveL, fluxL = self.Util.resample(wave, flux, step=step)  
        print(waveL.shape, fluxL.shape, error.shape, para.shape)

        SAVE_PATHL = SAVE_DIR + nameL
        self.save_dataset(waveL, fluxL, error, para, snr, SAVE_PATHL)




    
    def step5_convert(self, R=None, N=None, grid=0, step=20):
        RList= self.c.Rnms if R is None else [R]
        for R in RList:
            self.convert(R, N=N, step=step, grid=grid)

    def step6_PCA(self, W="RML", N=1000, top=100, name=""):
        p = BasePCA()
        p.run(W, N=N, top=top, transform=0, save=1, name=name)
        

    # def convert_v0(self, R, W="RedM", N=1000, boszR=5000, step=20, grid=0):
    #     wave, flux, error, para, snr = self.load_dataset(R, W=W, N=N, grid=0)  
    #     RR = self.c.dRR[R]
    #     nn = N // 1000
    #     SAVE_DIR = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/"
    #     if not os.path.isdir(SAVE_DIR):
    #         os.mkdir(SAVE_DIR)
    #     SAVE_PATH = SAVE_DIR + f"{W}_R{self.pixelR}_{nn}k{self.mag_name}.h5"
        
    #     self.save_dataset(wave, flux, error, para, snr, SAVE_PATH)    
    #     waveL, fluxL = self.Util.resample(wave, flux, step=step)  
    #     print(flux.shape, fluxL.shape, error.shape, para.shape)
    #     w = self.c.dWw[W][1]
    #     ws = self.c.dWs[w]
    #     SAVE_PATHL = SAVE_DIR + f"{ws[3]}_R{ws[2]}_{nn}k{self.mag_name}.h5"
    #     self.save_dataset(waveL, fluxL, error, para, snr, SAVE_PATHL)