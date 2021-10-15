import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .dnn import DNN 
from tqdm import tqdm
from lv.constants import Constants as c
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class BaseDNN():
    def __init__(self):
        self.Rnms = c.Rnms
        self.nR = len(self.Rnms) 
        self.RRnms = c.RRnms
        self.Pnms = c.Pnms
        self.dWs = c.dWs
        self.dRs = c.dRs
        self.dRR = c.dRR
        self.dRC = c.dRC
        
        self.Ws = None
        self.Rs = None

        self.dnns = None
        self.x_trains = {}
        self.y_trains = {}
        # self.p_trains= {}
        self.f_tests = {}
        self.p_tests= {}
        self.x_tests = {}
        self.y_tests = {}
        self.p_preds = {}
        self.dp_preds = {}
        self.ns_preds = {}
        self.pdx = None
        self.pRanges = {}
        self.pMins = {}
        self.pMaxs = {}
        self.PCs = {}

        self.wave = None
        self.resolution = 1000

        
############################################ DATA #######################################
    
    def load_R_dataset(self, R, W="RedM",N=10000, step=None):
        if step is None: step = self.step
        nn=N // 1000
        RR = self.dRR[R]
        DATA_PATH = f"/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset/{RR}/bosz_5000_{W}_{nn}k/dataset.h5"  
        wave, flux, error, para, dfsnr = self.load_dataset(DATA_PATH)
        if step is not None:
            wave, flux = self.resample(wave, flux, step=step, verbose=1)
        return wave, flux, error, para, dfsnr


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

    def load_RBF_data(self, N, R=None):
        nn= N // 1000
        DATA_PATH = f"/scratch/ceph/swei20/data/dnn/{self.dR[R]}/rbf_R{self.resolution}_{nn}k.h5"
        with h5py.File(DATA_PATH, 'r') as f:
            wave = f['wave'][()]
            flux = f['logflux'][()]
            pval = f['pval'][()]
            
        # print(wave.shape, flux.shape, pval.shape)
        fluxs = {}
        for W in self.Wnms:
            Ws = self.dWs[W]
            _, fluxs[W] = self.get_flux_in_Wrange(wave, flux, Ws)   
        return fluxs, pval[:, self.pdx]

    ############################################ SCALER #######################################
    def setup_scalers(self):
        for R, Rs in self.dRs.items():
            self.pRanges[R], self.pMins[R], self.pMaxs[R] = self.get_scaler(Rs)

    def get_scaler(self, Rs0):
        Rs = np.array(Rs0).T
        Rs = Rs[:, self.pdx]
        return np.diff(Rs, axis=0)[0], Rs[0], Rs[1]

    def scale(self, pval, R):
        pnorm = (pval - self.pMins[R]) / self.pRanges[R]        
        return pnorm

    def rescale(self, pnorm, R):
        pval = pnorm * self.pRanges[R] + self.pMins[R]
        return pval

    ############################################ DNN ##########################################

    def prepare_DNN(self, lr=0.01, dp=0.0):
        dnn = DNN()
        dnn.set_model_shape(self.n_ftr, len(self.pdx))
        dnn.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        dnn.build_model()
        return dnn

    def predict(self, data, R, dnn=None):
        if dnn is None: dnn = self.dnns[R]
        y_preds = dnn.model.predict(data)
        return self.rescale(y_preds, R)


    def plot_nsbox_R0_R1(self, R0, R1, data=None, Ps=None, SN=None,  n_box=2, ylbl=1,  axs=None):
        if axs is None: axs = plt.subplots(1, self.npdx,  figsize=(16, 4), facecolor="w")[1]
        pRange, pMin, pMax, = self.pRanges[R0], self.pMins[R0], self.pMaxs[R0]
        if data is None: 
            data = self.p_preds[R0][R1] if SN is None else self.ns_preds[R0][R1]
            name =  f'{self.dR[R1]}_Pred ({100* self.dCT[R0][R1]:.1f}%)' if SN is None else f'{self.dR[R1]}_SN={SN}'
        for i, ax in enumerate(axs):
            j = 0 if i + 1 == 3 else i + 1

            ax.scatter(data[:,i], data[:,j],s=1, c=self.dRC[R1])
            # ax.annotate(f"{self.dR[R0]}-NN", xy=(0.5,0.8), xycoords="axes fraction",fontsize=15, c=self.dRC[R0])
            handles, labels = ax.get_legend_handles_labels()

            ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor="r",lw=2, facecolor="none"))
            legend_ele = [Line2D([0], [0], marker='o',color='w', label=name, markerfacecolor=self.dRC[R1], markersize=10)]

            if R0 != R1:
                # ax.scatter(self.p_preds[R0][R0][:,i],self.p_preds[R0][R0][:,j],s=1, c=self.dRC[R0], label= f"{self.dR[R0]}")
                ax.add_patch(Rectangle((self.pMins[R1][i],self.pMins[R1][j]),\
                    (self.pRanges[R1][i]),(self.pRanges[R1][j]),edgecolor="k",lw=2, facecolor="none"))
                legend_ele.append(Patch(facecolor='none', edgecolor='r', label=f"{self.dR[R0]}-NN")) 
                legend_ele.append(Patch(facecolor='none', edgecolor='k', label=f"{self.dR[R1]}"))
            else:
                legend_ele.append(Patch(facecolor='none', edgecolor='r', label=f"{self.dR[R0]}-NN") )

            ax.set_xlim(pMin[i]-n_box*pRange[i], pMax[i]+n_box*pRange[i])
            ax.set_ylim(pMin[j]-n_box*pRange[j], pMax[j]+n_box*pRange[j])
            ax.set_xlabel(self.Pnms[i])
            if Ps is not None: ax.set_title(f"[M/H] = {Ps[0]}, Teff={Ps[1]}K, logg={Ps[2]}")
            ax.legend(handles = legend_ele)
            if ylbl: ax.set_ylabel(self.Pnms[j])