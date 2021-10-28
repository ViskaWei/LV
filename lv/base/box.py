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
        self.boszR=50000
        self.pixelR={"RedM": 5000, "Blue":2300, "NIR": 4300}
        self.mag=19

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
        
        self.slurm = slurm + " -t 72:0:0"

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
        return MH, Teff, logG, CM, ALPHA, uM, uT, uG, uC, uA
        
#step1 Box ---------------------------------------------------------------------------------

    def setBlocks(self):
        Blocks = collections.namedtuple('Blocks',['let','name','lower','upper','color'])
        name  = ['M31 Giants','MW Warm MS','MW Cool MS','Blue HB','Red HB', 'Dwarf G Giants']
        lower, upper = [], []
        for R, bnds in self.c.dRs.items():
            bnds = np.array(bnds)[[1,2,0]].T
            bnds[:,0] /= 1000
            lower.append(list(bnds[0]))
            upper.append(list(bnds[1]))    
        return Blocks(self.c.Rnms, name, lower, upper, list(self.c.dRC.values()))



#step2 RBF -----

    def get_rbf_cmd(self, R=None, boszR=5000):
        print(R, sep="/n/n")
        pp = self.c.dRs[R][:3]
        base = f"./scripts/build_rbf.sh {self.slurm} grid bosz --config ./configs/import/stellar/bosz/rbf/"
        ins = f" --in /scratch/ceph/dobos/data/pfsspec/import/stellar/grid/bosz_{boszR}"
        out =  f" --out /scratch/ceph/swei20/data/pfsspec/import/stellar/rbf/bosz_{boszR}_{self.c.dRR[R]}/"
        param = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} "
        cmd = base + ins+ out + param
        if self.boszR is None: self.boszR=boszR
        print(cmd)
    
    def step2_RBF(self, mkdir=1, boszR=5000):
        if mkdir: 
            for R in self.c.Rnms:
                os.mkdir(f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{self.c.dRR[R]}/")  
        for R in self.c.Rnms:
            self.get_rbf_cmd(R, boszR=boszR)

# step3 GRID -------------------------------------------------

    def get_pca_cmd(self, R, W="RedM", pixelR=5000, mag=19):
        print(R, sep="/n/n")
        pp = self.c.dRs[R]
        w  = self.c.dWw[W][0]
        base = f"./scripts/prepare.sh {self.slurm} model bosz pfs --config ./configs/infer/pfs/bosz/nowave/prepare/train.json"
        arm  = f"  ./configs/infer/pfs/bosz/nowave/inst_pfs_{w}.json"
        size = f" --chunk-size 1 "
        inD  = f" --in /scratch/ceph/dobos/data/pfsspec/import/stellar/grid/bosz_50000"
        
        if self.grid_name is None: self.grid_name = f"R{pixelR}_{W}_m{mag}"
        outD = f" --out /scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{self.c.dRR[R]}/laszlo/{self.grid_name}"
        para = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} --C_M {pp[3][0]} {pp[3][1]} --O_M {pp[4][0]} {pp[4][1]}"
        norm = f" --norm none"
        mag  = f" --mag-filter /scratch/ceph/dobos/data/pfsspec/subaru/hsc/hsc_i.dat --mag {mag} "
        grid = f" --sample-mode grid"

        cmd = base + arm + size + inD + outD + para + norm + mag + grid
        print(cmd)

    def step3_grid(self, R=None, W="RedM", pixelR=None, mag=19):
        self.mag=mag
        self.grid_name = f"R{pixelR}_{W}_m{self.mag}"
        RList= self.c.Rnms if R is None else [R]
        for R in RList:
            self.get_pca_cmd(R, W=W, pixelR=pixelR, mag=mag)


    def get_sample_cmd(self, R, W="RedM", N=10000, pixelR=None, dmag=1, Ps_arm=None):
        if pixelR is None: pixelR=self.pixelR[W]
        print(R, sep="/n/n")
        pp = self.c.dRs[R]
        w  = self.c.dWw[W][0]
        nn = N // 1000 
        chunk=10 if N <1000 else 1000

        base = f"./scripts/prepare.sh {self.slurm} model bosz-rbf pfs --config ./configs/infer/pfs/bosz/nowave/prepare/train.json"
        arm  = f"  ./configs/infer/pfs/bosz/nowave/inst_pfs_{w}.json"
        size = f" --chunk-size {chunk} --sample-count {N}"
        inD  = f" --in /scratch/ceph/swei20/data/pfsspec/import/stellar/rbf/bosz_{self.boszR}_{self.c.dRR[R]}/rbf"
        sample_name = f"R{pixelR}_{W}_{nn}k_m{self.mag}"
        outD = f" --out /scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{self.c.dRR[R]}/laszlo/{sample_name}"
        para = f" --Fe_H {pp[0][0]} {pp[0][1]} --T_eff {pp[1][0]} {pp[1][1]} --log_g  {pp[2][0]} {pp[2][1]} --C_M {pp[3][0]} {pp[3][1]} --O_M {pp[4][0]} {pp[4][1]}"
        mag  = f" --mag-filter /scratch/ceph/dobos/data/pfsspec/subaru/hsc/hsc_i.dat --mag {self.mag-dmag} {self.mag+dmag}"
        norm = f" --norm none"
        cmd = base + arm + size + inD + outD + para + mag + norm
        if Ps_arm is not None:
            sample_name_params = f"R{pixelR}_{Ps_arm}_{nn}k_m{self.mag}"
            params = f" --match-params /scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{self.c.dRR[R]}/laszlo/{sample_name_params}/"
            cmd += params
        print(cmd)


    def step4_sample(self, W=None, R=None, N=1000, Ps_arm=None):
        RList= self.c.Rnms if R is None else [R]
        for R in RList:
            self.get_sample_cmd(R, W=W, N=N, pixelR=5000, Ps_arm=Ps_arm)

    def load_laszlo(self, R, W="RedM", N=None, DATA_PATH=None, grid=False):
        RR = self.c.dRR[R]
        if DATA_PATH is None: 
            DATA_DIR =f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/laszlo/"
            if grid: 
                name = f"R{self.pixelR[W]}_{W}_m{self.mag}/"
            else:
                name = f"R{self.pixelR[W]}_{W}_{N//1000}k_m{self.mag}/"
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
            f.create_dataset(f"flux", data=flux, shape=flux.shape)
            f.create_dataset(f"pval", data=para, shape=para.shape)
            f.create_dataset(f"wave", data=wave, shape=wave.shape)  
            f.create_dataset(f"error", data=error, shape=error.shape)          
            f.create_dataset(f"snr", data=snr, shape=snr.shape)


    
    def convert(self, R, W="RedM", N=None, step=20, grid=0, DATA_PATH=None):
        wave, flux, error, para, snr = self.load_laszlo(R, W=W, N=N, grid=grid, DATA_PATH=DATA_PATH)  
        RR = self.c.dRR[R]
        SAVE_DIR = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/"
        w = self.c.dWw[W][1]
        ws = self.c.dWs[w]
        if grid: 
            SAVE_DIR += f"grid/"
            name = f"{W}_R{self.pixelR[W]}_m{self.mag}.h5"
            nameL = f"{ws[3]}_R{ws[2]}_m{self.mag}.h5"
        else:
            SAVE_DIR += f"sample/"
            name = f"{W}_R{self.pixelR[W]}_{N//1000}k_m{self.mag}.h5"
            nameL = f"{ws[3]}_R{ws[2]}_{N//1000}k_m{self.mag}.h5"
        if not os.path.isdir(SAVE_DIR):
            os.mkdir(SAVE_DIR)

        SAVE_PATH = SAVE_DIR + name
        print(SAVE_PATH)
        self.save_dataset(wave, flux, error, para, snr, SAVE_PATH)    
        
        waveL, fluxL = self.Util.resample(wave, flux, step=step)  
        print(waveL.shape, fluxL.shape, error.shape, para.shape)

        SAVE_PATHL = SAVE_DIR + nameL
        print(SAVE_PATHL)
        self.save_dataset(waveL, fluxL, error, para, snr, SAVE_PATHL)


    def mkdir(self, name, ):
        for RR in self.c.RRnms:
            if not os.path.isdir(name):
                os.mkdir(name)
    # os.mkdir(f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/snr/")



    
    def step5_convert(self,  R=None, W="RedM",N=None, grid=0, step=20):
        RList= self.c.Rnms if R is None else [R]
        for R in RList:
            self.convert(R, W=W, N=N, step=step, grid=grid)

    def step6_PCA(self, W="RML", N=1000, top=200):
        p = BasePCA()
        p.run(W, N=N, top=top, transform=0, save=1)
        

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