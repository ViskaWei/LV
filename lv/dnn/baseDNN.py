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
from lv.util import Util as u
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
        self.dataDir = "/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset_5000/"
        self.Rnms = c.Rnms
        self.nR = len(self.Rnms) 
        self.RRnms = c.RRnms
        self.Pnms = c.Pnms
        self.dWs = c.dWs
        self.dRs = c.dRs
        self.dR = c.dRR
        self.dRC = c.dRC
        
        self.Ws = None
        self.Rs = None

        self.dnns = {}
        self.x_trains = {}
        self.y_trains = {}
        self.p_trains= {}
        self.f_trains = {}
        self.s_trains = {}

        self.f_tests = {}
        self.s_tests = {}
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
        self.dPC = {}
        self.dPxl={}
        self.dCT = {}

        self.wave = None
        self.resolution = 1000

        self.snrList=[10,20,30,50,100]
        

        
############################################ DATA #######################################
# data cleaning -------------------------------------------------------------    
    def get_snr(self, fluxs):
        if isinstance(fluxs, list) or (len(fluxs.shape)>1):
            SNs = []
            for nsflux in fluxs:
                SNs.append(u.getSN(nsflux))
            return np.mean(SNs)
        else:
            print("not list")
            return u.getSN(fluxs)

    def resample(self, wave, fluxs, step=20, verbose=1):
        return u.resample(wave, fluxs, step=step, verbose=verbose)

    def resampleFlux_i(self, flux, step=20):
        return u.resampleFlux_i(flux, step=step)

    def load_R_dataset(self, R, W="RedM",N=10000, step=None):
        if step is None: step = self.step
        nn=N // 1000
        RR = self.dR[R]
        DATA_PATH = f"{self.dataDir}/{RR}/bosz_5000_{W}_{nn}k/dataset.h5"  
        wave, flux, error, para, dfsnr = self.load_dataset(DATA_PATH)
        if step is not None:
            wave, flux = self.resample(wave, flux, step=step, verbose=1)
        return wave, flux, error, para, dfsnr

    # def load_dataset(self, DATA_PATH):
    #     with h5py.File(DATA_PATH, "r") as f:
    #         wave = f["wave"][()]
    #         flux = f["flux"][()]
    #         mask = f["mask"][()]
    #         error = f["error"][()]
    #     assert (np.sum(mask) == 0)
    #     dfparams = pd.read_hdf(DATA_PATH, "params")    
    #     para = dfparams[['Fe_H','T_eff','log_g','C_M','O_M']].values
    #     dfsnr = dfparams[['redshift','mag','snr']]
    #     return wave, flux, error, para, dfsnr

    def load_PC_W(self, W=None, Rs=None, top=100):
        if Rs is None: Rs = self.Rnms
        Ws = self.dWs[W]
        PC_PATH = f"/scratch/ceph/swei20/data/dnn/PC/logPC/{Ws[3]}_R{Ws[2]}.h5"
        dPC = {}
        with h5py.File(PC_PATH, 'r') as f:
            for R in Rs:
                PC = f[f'PC_{R}'][()]
                dPC[R] = PC[:top]
        nPixel = PC.shape[1]        
        return dPC, nPixel

    def load_RBF_W_R(self, W="RML", R=None, N=1000):
        nn= N // 1000
        RR = self.dR[R]
        Ws = self.dWs[W]
        DATA_PATH = f"{self.dataDir}/{RR}/{Ws[3]}_R{Ws[2]}_{nn}k.h5"
        with h5py.File(DATA_PATH, 'r') as f:
            wave = f['wave'][()]
            flux = f['logflux'][()]
            pval = f['pval'][()]
            error = f['error'][()]
            snr =   f['snr'][()]
        if self.pdx is not None: pval = pval[:,self.pdx]
        return wave, flux, error, pval, snr

    def load_snr_flux(self, W, R0, SNR_PATH=None, idx=0):
        RR = self.dR[R0]
        if SNR_PATH is None: SNR_PATH=f"{self.dataDir}/{RR}/SNR/{W}_{idx}.h5"    
        with h5py.File(SNR_PATH, "r") as f:
            wave = f["wave"][:]
            flux = f["flux"][:]
            para = f["para"][:]
        dSnr={}
        with h5py.File(SNR_PATH, "r") as f:
            for snr in self.snrList:
                dSnr[snr] = f[f"snr_{snr}"][()]
        return wave, flux, para, dSnr

    def prepare_snr_flux(self, R0, W="RedM", N=100, plot=1):
        wave, flux, para, dSnr = self.load_snr_flux(W, R0)
        nsfluxs = self.generate_nsflux_snr(flux, N, dSnr)
        dStats={}
        # w = self.dWw[W]
        for snr in dSnr.keys():
            dStats[snr] = self.eval_nsflux_snr(nsfluxs[snr], "RML", R0)
        means = [dStats[snr]["mean"] for snr in self.snrList]
        varss = [dStats[snr]["var"] for snr in self.snrList]
        dStats["mean"] = means
        dStats["var"]=varss
        if plot:
            self.plot_snr(dStats, W, R0)
        return dStats

    def plot_snr(self, dStats, W, R0, ax=None):
        if ax is None: fig, ax = plt.subplots(figsize=(5,4), facecolor='w')
        ax.plot(self.snrList, dStats["mean"], 'o-', label="mean")
        ax.plot(self.snrList, dStats["var"], 'o-', label="std")
        ax.set_xticks(self.snrList)
        ax.set_xlabel("SNR")
        ax.set_ylabel(self.dR[R0])
        ax.set_title(W)
        ax.legend()


    def generate_nsflux_snr(self, flux, N, dSnr):
        dSNFlux={}
        for snr, err in dSnr.items():
            fluxL = self.resampleFlux_i(flux,step=20)
            dSNFlux[snr] = self.add_noise_N(fluxL, dSnr[snr], N=N)
        return dSNFlux
    
    def eval_nsflux_snr(self, nsflux_snr, W, R, norm=1):
        if norm:
            preds = self.trans_predict_norm(nsflux_snr, W, R)
        else:
            preds = self.trans_predict(nsflux_snr, W, R)
        mean = preds.mean(axis=0)
        centered = preds - mean
        # cov = centered.T.dot(centered)
        cov = centered
        u,s,v = np.linalg.svd(cov)
        stats={}
        stats["mean"], stats["var"]=s.mean(), s.std()
        stats["v"] = v
        return stats

    def trans_predict_norm(self, x, W, R, dnn=None):
        data = self.transform_W_R(x, W, R)
        if dnn is None: dnn = self.dnns[R]
        y_preds = dnn.model.predict(data)
        return y_preds





#-----------------------------------------------
    def process_RBF_W_R(self, W, R, N=1000, isNoisy=1):
        _, flux, error, pval, snr = self.load_RBF_W_R(W=W, R=R, N=N)
        if isNoisy:
            flux = self.add_noise(flux, error)
        return flux, pval, snr
    
    def prepare_testset_W(self, W, N_test, isNoisy=1):
        for R0 in self.Rnms:
            self.f_tests[R0],self.p_tests[R0], self.s_tests[R0] =self.process_RBF_W_R(W, R0, N_test, isNoisy=isNoisy)
        for R0 in self.Rnms:
            x = {}
            for R1 in self.Rnms:
                x[R1] = self.transform_W_R(self.f_tests[R1], W, R0) # project to R0 PC
            self.x_tests[R0] = x


    def prepare_trainset_W(self,W, N_train, isNoisy=1):
        for R0 in self.Rnms:
            self.f_trains[R0], self.p_trains[R0], self.s_trains[R0] = self.process_RBF_W_R(W, R0, N=N_train, isNoisy=isNoisy)
            self.x_trains[R0] = self.transform_W_R(self.f_trains[R0], W, R0) # project to R0 PC
            self.y_trains[R0] = self.scale(self.p_trains[R0], R0)

# noise ---------------------------------------------------------------------------------
    def add_noise(self, fluxs, errs,  rate=1.0, step=20):
        nsflux = np.zeros_like(fluxs)
        for ii, flux in enumerate(fluxs):
            noiseL = self.get_noise_i(flux, errs[ii], rate=rate, step=step)
            nsflux[ii]= flux + noiseL
        return nsflux

    def get_noise_i(self, flux, err, rate=1.0, step=20):
        noise = rate * np.random.normal(0, err)
        noiseL = self.resampleFlux_i(noise, step=step)
        return noiseL

    def add_noise_N(self, flux, err, N, rate=1.0, step=20):
        nsflux = np.zeros((N, flux.shape[0]))
        for i in range(N):
            nsflux[i] = flux + self.get_noise_i(flux, err, rate=rate, step=step)
        return nsflux

    

# scaler ---------------------------------------------------------------------------------
    def setup_scalers(self, pdx):
        self.pdx = pdx
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

    def transform_W_R(self, x, W, R):
        return x.dot(self.dPC[W][R].T)
  


    ############################################ DNN ##########################################

    def prepare_DNN(self, input_dim=None, lr=0.01, dp=0.0):
        dnn = DNN()
        if input_dim is None: input_dim = self.n_ftr
        dnn.set_model_shape(input_dim, len(self.pdx))
        dnn.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        dnn.build_model()
        return dnn

    def predict(self, data, R, dnn=None):
        if dnn is None: dnn = self.dnns[R]
        y_preds = dnn.model.predict(data)
        return self.rescale(y_preds, R)

    def trans_predict(self, data, W, R0, dnn=None):
        pc_data = self.transform_W_R(data, W, R0)
        p_pred = self.predict(pc_data, R0, dnn=None)
        return p_pred

    def plot_box_R0_R1(self, R0, R1, data=None, Ps=None, SN=None,  n_box=2, ylbl=1,  axs=None):
        if axs is None: axs = plt.subplots(1, self.npdx,  figsize=(16, 4), facecolor="w")[1]
        pRange, pMin, pMax, = self.pRanges[R0], self.pMins[R0], self.pMaxs[R0]
        if data is None: 
            data = self.p_preds[R0][R1] if SN is None else self.ns_preds[R0][R1]
            name =  f'{self.dR[R1]}_Pred ({100* self.dCT[R0][R1]:.1f}%)' if SN is None else f'{self.dR[R1]}_SN={SN:.2f}'
        else:
            name = f'{self.dR[R0]}-NN'
        for i, ax in enumerate(axs):
            j = 0 if i + 1 == 3 else i + 1

            ax.scatter(data[:,i], data[:,j],s=1, c=self.dRC[R1])
            # ax.annotate(f"{self.dR[R0]}-NN", xy=(0.5,0.8), xycoords="axes fraction",fontsize=15, c=self.dRC[R0])
            handles, labels = ax.get_legend_handles_labels()

            legend_ele = [Line2D([0], [0], marker='o',color='w', label=name, markerfacecolor=self.dRC[R1], markersize=10)]

            if R0 != R1:
                # ax.scatter(self.p_preds[R0][R0][:,i],self.p_preds[R0][R0][:,j],s=1, c=self.dRC[R0], label= f"{self.dR[R0]}")
                ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor=self.dRC[R0],lw=2, facecolor="none"))
                ax.add_patch(Rectangle((self.pMins[R1][i],self.pMins[R1][j]),\
                    (self.pRanges[R1][i]),(self.pRanges[R1][j]),edgecolor="k",lw=2, facecolor="none"))
                legend_ele.append(Patch(facecolor='none', edgecolor=self.dRC[R0], label=f"{self.dR[R0]}-NN")) 
                legend_ele.append(Patch(facecolor='none', edgecolor='k', label=f"{self.dR[R1]}"))
            else:
                ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor="k",lw=2, facecolor="none"))
                legend_ele.append(Patch(facecolor='none', edgecolor='k', label=f"{self.dR[R0]}-NN") )

            ax.set_xlim(pMin[i]-n_box*pRange[i], pMax[i]+n_box*pRange[i])
            ax.set_ylim(pMin[j]-n_box*pRange[j], pMax[j]+n_box*pRange[j])
            ax.set_xlabel(self.Pnms[i])
            if Ps is not None: ax.set_title(f"[M/H] = {Ps[0]:.2f}, Teff={int(Ps[1])}K, logg={Ps[2]:.2f}")
            ax.legend(handles = legend_ele)
            if ylbl: ax.set_ylabel(self.Pnms[j])

    def plot_SNR_R0(self, R0, data=None, Ps=None, SN=None,  n_box=2, ylbl=1,  axs=None):
        R1 = R0
        if data is None: 
            data = self.ns_preds[R0][R0]
            name = f'{self.dR[R1]}_SN={SN:.2f}'
        if axs is None: axs = plt.subplots(1, self.npdx,  figsize=(16, 4), facecolor="w")[1]
        pRange, pMin, pMax, = self.pRanges[R0], self.pMins[R0], self.pMaxs[R0]

 
        for i, ax in enumerate(axs):
            j = 0 if i + 1 == 3 else i + 1

            ax.scatter(data[:,i], data[:,j],s=1, c=self.dRC[R1])
            # ax.annotate(f"{self.dR[R0]}-NN", xy=(0.5,0.8), xycoords="axes fraction",fontsize=15, c=self.dRC[R0])
            handles, labels = ax.get_legend_handles_labels()

            ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor="r",lw=2, facecolor="none"))
            legend_ele = [Line2D([0], [0], marker='o',color='w', label=name, markerfacecolor=self.dRC[R1], markersize=10)]
            legend_ele.append(Patch(facecolor='none', edgecolor='r', label=f"{self.dR[R0]}-NN") )

            ax.set_xlim(pMin[i]-n_box*pRange[i], pMax[i]+n_box*pRange[i])
            ax.set_ylim(pMin[j]-n_box*pRange[j], pMax[j]+n_box*pRange[j])
            ax.set_xlabel(self.Pnms[i])
            if Ps is not None: ax.set_title(f"[M/H] = {Ps[0]:.2f}, Teff={int(Ps[1])}K, logg={Ps[2]:.2f}")
            ax.legend(handles = legend_ele)
            if ylbl: ax.set_ylabel(self.Pnms[j])

    def plot_box_R0(self, R0, data=None, Ps=None, SN=None, n_box=2,  axs=None, large=0):
        if axs is None: 
            if large:
                f, axss = plt.subplots(self.nR,self.npdx, figsize=(16, 4*self.nR), sharey="col", sharex="col", facecolor="w")
            else:
                f, axss = plt.subplots(self.npdx, self.nR, figsize=(20, 4*self.npdx), sharey="row", sharex="row", facecolor="w")
                axss = axss.T
                
        for i, axs in enumerate(axss):
            R1 = self.Rnms[i]
            self.plot_box_R0_R1(R0, R1, data=data, Ps=Ps, SN=SN, n_box=n_box, axs=axs, ylbl = (i==0))
        # plt.tight_layout()

    def plot_pred(self, R0):
        l = len(self.pdx)
        f, axs = plt.subplots(1, l,figsize=(10,4), facecolor="w")
        for ii, ax in enumerate(axs):
            ax.scatter(self.p_tests[R0][:,ii], self.p_preds[R0][R0][:,ii], c="k",s=1) #, label=f"{self.Pnms[pdx]}"
            # if R is not None:
                # axs[0][ii].scatter(self.YOs[R][:,ii], self.YPs[R][:,ii], c="b",s=1, label=f"{R}")
            ax.plot([self.pMins[R0][ii], self.pMaxs[R0][ii]], [self.pMins[R0][ii], self.pMaxs[R0][ii]], c="r", lw=2)
            ax.annotate(f"{self.dR[R0]}-NN\n{self.Pnms[ii]}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=20)
            # axs[1][ii].plot(np.array([[self.pMin[ii],self.pMin[ii]], [self.pMax[ii]],self.pMax[ii]]), c="r")
            ax.set_xlim(self.pMins[R0][ii], self.pMaxs[R0][ii])
            ax.set_ylim(self.pMins[R0][ii], self.pMaxs[R0][ii])
            # ax.legend()
            # axs[1][ii].scatter(self.y_test_org[:,ii], self.dy_pred[:,ii], c="k",s=1, label=f"{self.R}")
            # if R is not None:
                # axs[1][ii].scatter(self.YOs[R][:,ii], self.dYPs[R][:,ii], c="b",s=1, label=f"{R}")

            # axs[1][ii].annotate(f"{self.R}-NN\n{c.Pnms[ii]}", xy=(0.6,0.2), xycoords="axes fraction", fontsize=20)
            # axs[1][ii].axhline(0, c='r')
            # axs[1][ii].legend()
        axs[0].set_ylabel(f"pred")
        # axs[1][0].set_ylabel(f"$\Delta$pred")
########################################## overlap ##########################################
    def get_overlap_R0_R1(self, R0, R1):
        p_pred = self.p_preds[R0][R1]
        mask = True
        for pdx in range(self.npdx):
            mask = mask & (p_pred[:,pdx] >= self.pMins[R0][pdx]) & (p_pred[:,pdx] <= self.pMaxs[R0][pdx])
        overlap = mask.sum() / self.N_test
        return overlap

    def get_overlap_R0(self, R0):
        overlaps = {}
        for R1 in self.Rnms:
            overlaps[R1] = self.get_overlap_R0_R1(R0, R1)
        return overlaps

    def get_overlap(self):
        overlaps = {}
        for R0 in self.Rnms:
            overlaps[R0] = self.get_overlap_R0(R0)
        return overlaps

    def get_overlap_mat(self, plot=1):
        CT = np.zeros((len(self.Rnms), len(self.Rnms)))
        for ii, R0 in enumerate(self.Rnms):
            for jj, R1 in enumerate(self.Rnms):
                # if ii == jj:
                #     CT[ii,jj] = 1 - self.get_overlap_R0_R1(R0, R1)
                # else:
                CT[ii][jj] = self.get_overlap_R0_R1(R0, R1)

        self.CT = CT
        if plot: self.plot_heatmaps()

    def plot_heatmap_v1(self, size=2000, cut=0.005, ax=None):
        nn = len(self.Rnms)
        xv, yv = np.meshgrid(np.arange(nn), np.arange(nn))
        yv = np.flipud(yv)
        if ax is None: 
            f, ax = plt.subplots(figsize=(5,5)) #, facecolor="gray"
        # ax.set_facecolor('lightgray')
        mat = self.CT - np.diag(np.diag(self.CT))
        vmax = np.max(mat)
        ax.scatter(x=xv, y=yv, s=size / vmax * mat, marker='s', c = np.log(mat), cmap="autumn")
        ax.grid(False, 'major')
        ax.grid(True, 'minor', color='w',linewidth=2)
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()[:-1]], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()[:-1]], minor=True)
        mask = mat > cut
        xvm = xv[mask]
        yvm = yv[mask]
        ctm = mat[mask]
        for ii in range(mask.sum()):            
            ax.annotate(f"{ctm[ii]}", xy=(xvm[ii], yvm[ii]), xycoords="data", fontsize=15, ha='center')
        
        xlbl = [self.RRnms[0], *self.RRnms]
        ax.set_xticklabels(xlbl)
        ylbl = [self.RRnms[0], *self.RRnms[::-1]]
        ax.set_yticklabels(ylbl, rotation=90,  verticalalignment='center', horizontalalignment='right')
        ax.set_title(f"Error > {100*cut}%")

    def plot_heatmap(self, ax=None):
        if ax is None:
            f, ax = plt.subplots(figsize=(6,5), facecolor="gray")
        sns.heatmap(self.CT, vmax=0.1, ax=ax, annot=True, cmap="inferno")
        ax.set_xticklabels(self.RRnms)
        ax.set_yticklabels(self.RRnms)
        ax.set_title("overlap Heatmap")

    def plot_heatmaps(self):
        plt.style.use('seaborn-darkgrid')
        f, axs= plt.subplots(1,2,figsize=(12,5), facecolor="w", gridspec_kw={'width_ratios': [6, 5]})
        self.plot_heatmap(ax=axs[0])
        self.plot_heatmap_v1(ax=axs[1])

