import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .dnn import DNN 
from tqdm import tqdm
from scipy.stats import chi2 as chi2

from lv.constants import Constants as c
from lv.util import Util
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle, Ellipse, Patch
import matplotlib.transforms as transforms

from matplotlib.lines import Line2D
from matplotlib import collections  as mc


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class BaseDNN():
    def __init__(self):
        self.dataDir = "/scratch/ceph/swei20/data/pfsspec/train/pfs_stellar_model/dataset"
        self.Rnms = c.Rnms
        self.nR = len(c.Rnms) 
        self.RRnms = c.RRnms
        self.Pnms = c.Pnms
        self.dWs = c.dWs
        self.dRs = c.dRs
        self.dR = c.dRR
        self.dRC = c.dRC
        self.dWw = c.dWw
        self.Util = Util()
        self.c = c()

        self.mag=19
        
        self.Ws = None
        self.Rs = None

        self.dnns = {}
        self.x_trains = {}
        self.y_trains = {}
        self.p_trains= {}
        self.f_trains = {}
        self.s_trains = {}

        self.o_tests=  {}
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
        self.dSNs = {}
        self.dstep=20
        self.N_snr=10
        self.N_fluxNL=100
        self.dSR={}
        self.dSN = {}
        self.dSN_idxs={}
        self.dSN_paras={}

        self.wave = None
        self.resolution = 1000
        self.dWRt = {"RML": 1, "BL":1}


        self.snrList=[10,20,30,50,100]
        self.debug=0
        


    def get_random_params(self, R, N, npdx=3):
        Rs = self.dRs[R][:npdx]
        out = np.zeros((N, npdx))
        for pdx in range(npdx):
            out[:, pdx] = np.random.uniform(Rs[pdx][0], Rs[pdx][1], N)
        return out
        
############################################ DATA #######################################
# data cleaning -------------------------------------------------------------    
    def get_snr(self, fluxs):
        if isinstance(fluxs, list) or (len(fluxs.shape)>1):
            SNs = []
            for nsflux in fluxs:
                SNs.append(self.Util.getSN(nsflux))
            return np.mean(SNs)
        else:
            print("not list")
            return self.Util.getSN(fluxs)

    def resample(self, wave, fluxs, step=20, verbose=1):
        return self.Util.resample(wave, fluxs, step=step, verbose=verbose)

    def resampleFlux_i(self, flux, step=20):
        return self.Util.resampleFlux_i(flux, step=step)

# snr -------------------------------------------------------------
    def init_snr_R(self, R0, W="RedM"):
        np.random.seed(0)
        self.load_dSN_W(W, R0)
        WL = self.dWw[W][1]
        dSN_preds = self.process_snr_flux(WL=WL, R0=R0)
        self.dSR[R0] = dSN_preds


    def load_dSN_W_R0(self, W, R0, SNR_PATH=None):
        RR = self.dR[R0]
        if SNR_PATH is None: SNR_PATH=f"{self.dataDir}/{RR}/snr/{W}_{self.N_snr}.h5"    
        dSN={}
        with h5py.File(SNR_PATH, "r") as f:
            for snr in self.snrList:
                dSN[snr] = f[f"snr_{snr}"][()]
        return dSN

    def load_dSN_W(self, W, Rs=None, SNR_PATH=None):
        Rs = self.Rnms if Rs is None else [Rs]
        for R0 in Rs:
            idxs=np.arange(self.N_snr)
            # idxs = np.random.randint(0, self.N_test, self.N_snr)
            self.dSN_idxs[R0] = idxs
            self.dSN[R0] = self.load_dSN_W_R0(W, R0, SNR_PATH)
            self.dSN_paras[R0] = self.p_tests[R0][idxs]
            
    def prepare_snr_flux(self, R0=None):
        dN_fluxs = {}
        for ii in range(self.N_snr):
            dN_fluxs[ii]= self.prepare_snr_flux_i(ii, R0=R0)
        return dN_fluxs
 
    def prepare_snr_flux_i(self, i, R0=None):
        idx = self.dSN_idxs[R0][i]
        dSN_fluxs_i = {}
        fluxL = self.o_tests[R0][idx]
        for snr in self.snrList:
            error = self.dSN[R0][snr][idx]
            fluxLs = self.add_noise_N(fluxL, error, self.N_fluxNL)
            lognorm_fluxLs = self.Util.lognorm_flux(fluxLs)
            dSN_fluxs_i[snr] = lognorm_fluxLs
        return dSN_fluxs_i

    def process_snr_flux_i(self, i, WL="RML", R0=None):
        dSN_fluxs_i = self.prepare_snr_flux_i(i, R0=R0)
        dSN_preds_i = {}
        for snr, fluxNLs in dSN_fluxs_i.items():
            dSN_preds_i[snr] = self.trans_predict(fluxNLs, W=WL, R0=R0)
        return dSN_preds_i

    def process_snr_flux(self, WL="RML", R0=None):
        dN_fluxs = self.prepare_snr_flux(R0)
        dSN_preds={}
        for ii, dSN_fluxs_i in dN_fluxs.items():
            dSN_preds_i={}
            for snr, fluxNLs in dSN_fluxs_i.items():
                dSN_preds_i[snr] = self.trans_predict(fluxNLs, W=WL, R0=R0)
            dSN_preds[ii] = dSN_preds_i
        return dSN_preds


    def plot_snr(self, dStats, W, R0, ax=None):
        if ax is None: fig, ax = plt.subplots(figsize=(5,4), facecolor='w')
        ax.plot(self.snrList, dStats["mean"], 'o-', label="mean")
        ax.plot(self.snrList, dStats["var"], 'o-', label="std")
        ax.set_xticks(self.snrList)
        ax.set_xlabel("SNR")
        ax.set_ylabel(self.dR[R0])
        ax.set_title(W)
        ax.legend()

    def eval_snr(self, R0, snr, N=None, n_box=0.2):
        if N is None: N = self.N_snr
        ffs=[]
        legend=1
        for i in range(N):
            preds = self.dSR[R0][i][snr]
            para = self.dSN_paras[R0][i]
            ffs = ffs + self.flow_fn_i(preds, para, snr=snr, legend=legend)
            legend=0
        self.plot_box_R0_R1(R0,R0, ffs, n_box=n_box)

    def eval_snr_i(self, i, R0, n_box=0.2):
        dSN_preds_i = self.dSR[R0][i]
        f, axs = plt.subplots(len(self.snrList), self.npdx, figsize=(5*self.npdx,5*len(self.snrList)), facecolor='w')
        
        for ii, (snr, preds_i) in enumerate(dSN_preds_i.items()):
            para = self.dSN_paras[R0][i]
            ffs = self.flow_fn_i(preds_i, para, snr=snr, legend=1)
            self.plot_box_R0_R1(R0,R0, ffs, n_box=n_box, axs=axs[ii])



#dataloader-----------------------------------------------
    def pcloader_W(self, W=None, Rs=None, top=None, name=""):
        if Rs is None: Rs = self.Rnms
        if top is None: top = self.top
        Ws = self.dWs[W]
        PC_PATH = f"/scratch/ceph/swei20/data/dnn/PC/logPC/{Ws[3]}_R{Ws[2]}{name}.h5"
        dPC = {}
        with h5py.File(PC_PATH, 'r') as f:
            for R in Rs:
                PC = f[f'PC_{R}'][()]
                dPC[R] = PC[:top]
        nPixel = PC.shape[1]        
        return dPC, nPixel

    def dataloader_W_R(self, W="RML", R=None, N=None, mag=None, grid=0):
        DATA_PATH = self.get_datapath_from_W_R(W, R, N)
        if self.debug: print(DATA_PATH)
        wave, flux, pval, error, snr = self.dataloader(DATA_PATH)
        if grid and (N is not None): 
            idx = np.random.choice(len(flux), N)
            flux, pval, error, snr = flux[idx], pval[idx], error[idx], snr[idx]
        return wave, flux, error, pval, snr

    def get_datapath_from_W_R(self, W, R, N, mag=None, grid=0):
        if mag is None: mag = self.mag
        Ws = self.dWs[W]
        if grid:
            DATA_PATH = f"{self.dataDir}/{self.dR[R]}/grid/{Ws[3]}_R{Ws[2]}_m{mag}.h5"
        elif not grid:
            nn= N // 1000
            DATA_PATH = f"{self.dataDir}/{self.dR[R]}/sample/{Ws[3]}_R{Ws[2]}_{nn}k_m{mag}.h5"
        return DATA_PATH

    def check_para(self, PATH1, PATH2):
        with h5py.File(PATH1, 'r') as f:
            para1 = f['pval'][()]
        with h5py.File(PATH2, 'r') as f:
            para2 = f['pval'][()]
        return abs(para1  - para2).sum()
        
    def dataloader(self, DATA_PATH):
        with h5py.File(DATA_PATH, 'r') as f:
            wave = f['wave'][()]
            flux = f['flux'][()]
            pval = f['pval'][()]
            error = f['error'][()]
            snr =   f['snr'][()]
        if self.pdx is not None: pval = pval[:,self.pdx]
        return wave, flux, pval, error, snr
    
    def dataloader_lazy(self, DATA_PATH):
        with h5py.File(DATA_PATH, 'r') as f:
            flux = f['flux'][()]
            error = f['error'][()]
        return flux, error
    

    def process_data_W_R(self, W, R, N=None, grid=0, mag=None, isNoisy=1, isLazy=0):
        if isLazy:
            DATA_PATH = self.get_datapath_from_W_R(W, R, N, mag=mag, grid=grid)
            if self.debug: print("L", DATA_PATH)
            fluxLs, error = self.dataloader_lazy(DATA_PATH)
            if isNoisy: fluxLs = self.add_noise(fluxLs, error, rate=self.dWRt[W])
            lognorm_fluxLs = self.Util.lognorm_flux(fluxLs)
            return lognorm_fluxLs
        else:
            _, fluxLs, error, pval, snr = self.dataloader_W_R(W=W, R=R, N=N, grid=grid, mag=mag)
            if isNoisy: fluxLs = self.add_noise(fluxLs, error, rate=self.dWRt[W])
            lognorm_fluxLs = self.Util.lognorm_flux(fluxLs)
            return fluxLs, lognorm_fluxLs, pval, snr

    def process_data_R(self, R, Ws=None, N=None, grid=0, isNoisy=1):
        if Ws is None: Ws = self.arms
        fluxLs, lognorm_fluxLs_W, pval, snr = self.process_data_W_R(Ws[0], R, N=N, grid=grid, isNoisy=isNoisy)
        if len(Ws) > 1:
            lognorm_fluxLs = [lognorm_fluxLs_W]
            for W in Ws[1:]:
                lognorm_fluxLs_W = self.process_data_W_R(W, R, N=N, grid=grid, isNoisy=isNoisy, isLazy=1)
                lognorm_fluxLs.append(lognorm_fluxLs_W)
            x = np.hstack(lognorm_fluxLs)
        else:
            x = lognorm_fluxLs_W
        return x, pval, snr


    
    
    def prepare_testset(self, Ws=None, Rs=None, N_test=None, grid=0, isNoisy=1):
        if Rs is None: Rs = self.Rnms
        for R0 in Rs:
            self.f_tests[R0], self.p_tests[R0], self.s_tests[R0] = self.process_data_R(R0, Ws=Ws, N=N_test, grid=grid, isNoisy=isNoisy)
        for R0 in Rs:
            pcFlux = {}
            for R1 in Rs:
                pcFlux[R1] = self.transform_R(self.f_tests[R1], R0) # project to R0 PC
            self.x_tests[R0] = pcFlux


    def prepare_trainset(self, Ws=None, Rs=None, N_train=None, grid=0, isNoisy=1):
        if Rs is None: Rs = self.Rnms
        for R0 in Rs:
            self.f_trains[R0], self.p_trains[R0], self.s_trains[R0] = self.process_data_R(R0, Ws=Ws, N=N_train, grid=grid, isNoisy=isNoisy)
            self.y_trains[R0] = self.scale(self.p_trains[R0], R0)
            self.x_trains[R0] = self.transform_R(self.f_trains[R0],R0) # project to R0 PC
                
                

# noise ---------------------------------------------------------------------------------
    def add_noise(self, fluxLs, errs,  rate=1.0, step=20):
        fluxNLs = np.zeros_like(fluxLs)
        for ii, fluxL in enumerate(fluxLs):
            noiseL = self.get_noise_i(errs[ii], rate=rate, step=step)
            fluxNLs[ii]= fluxL + noiseL
        return fluxNLs

    def get_noise_i(self, err, rate=1.0, step=20):
        noise = rate * np.random.normal(0, err)
        noiseL = self.resampleFlux_i(noise, step=step)
        return noiseL

    def add_noise_N(self, flux, err, N, rate=1.0, step=20):
        nsflux = np.zeros((N, flux.shape[0]))
        for i in range(N):
            nsflux[i] = flux + self.get_noise_i(err, rate=rate, step=step)
        return nsflux

    

# scaler ---------------------------------------------------------------------------------
    def setup_scalers(self, pdx=None):
        if pdx is None: pdx = self.pdx
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

    def transform_R(self, x, R):
        start=0
        y= []
        for ii, (W, pixel) in enumerate(self.dPxl.items()):
            end = start + pixel
            y.append(self.transform_W_R(x[:, start: end], W, R))
            start = end
        return np.hstack(y)


    def trans_predict_norm(self, x, W, R, dnn=None):
        data = self.transform_W_R(x, W, R)
        if dnn is None: dnn = self.dnns[R]
        y_preds = dnn.model.predict(data)
        return y_preds
  



#plot ellipse ---------------------------------------------------------------------------------

    # def process_fluxNL(self, fluxNL, W=None, R0=None, plot=1):
    #     preds=self.trans_predict(fluxNL, W, R0)
    #     if plot:
    #         preds_fn = self.scatter_fn(preds, c=self.dRC[R0], alpha=0.3)
    #         ellipse_fn = self.get_ellipse_fn(preds)
    #         return preds, [preds_fn, ellipse_fn]
    #     return preds

    def get_preds_stats(self, preds):
        dStats = {}
        mu, sigma = preds.mean(0), preds.std(0)
        centered = preds - mu

        dStats["mu"] = mu
        dStats["sigma"] = sigma


    def flow_fn(self,paras, center, legend=0):
        fpara=self.scatter_fn(paras, c="r",s=10)
        fmean=self.scatter_fn(center, c="g",s=10)
        ftraj=self.traj_fn(paras, center, c="r",lw=2)
        return [fpara,fmean, ftraj]

    def flow_fn_i(self, pred, center, snr=None, legend=0):
        mu, sigma = pred.mean(0), pred.std(0)
        center = np.array([center])
        MU = np.array([mu])
        lgd =f"SNR={snr}" if legend else None
        
        fpred=self.scatter_fn(pred, c="gray",s=10, lgd=lgd)
        fmean=self.scatter_fn(MU,  c="r",s=10)
        ftarget=self.scatter_fn(center,  c="g",s=10)
        ftraj=self.traj_fn(MU, center, c="r",lw=2)
        add_ellipse = self.get_ellipse_fn(pred,c='b', legend=legend)


        return [fpred,fmean, ftarget,ftraj, add_ellipse]

    def traj_fn(self, strts, ends, c=None, lw=2):
        nTest=strts.shape[0]
        def fn(i, j , ax, handles=[]):
            flowList=[]
            for ii in range(nTest):
                strt=strts[ii]
                end= ends[ii]
                flowList.append([(strt[i],strt[j]), (end[i],end[j])])
            lc = mc.LineCollection(flowList, colors=c, linewidths=lw)
            ax.add_collection(lc)
            return handles
        return fn

    def scatter_fn(self, data, c=None, s=1, lgd=None):
        def fn(i, j, ax, handles=[]):
            ax.scatter(data[:,i], data[:,j],s=s, c=c)
            if lgd is not None: 
                handles.append(Line2D([0], [0], marker='o',color='w', label=lgd, markerfacecolor=c, markersize=10))
            return handles
        return fn

    def SN_pred_fn(self, preds, snr=100, c=None):
        fns=[]
        for num, pred in preds.items():
            fns.append(self.scatter_fn(pred[snr], c=c))
        return fns


    def get_ellipse_params(self, pred):
        x0s,y0s,s05s,degs = [],[],[],[]
        for ii in range(self.npdx):
            jj = 0 if ii ==self.npdx-1 else ii + 1
            x0, y0, s05, degree = self.get_ellipse_param(pred[:,ii], pred[:,jj])
            x0s.append(x0)
            y0s.append(y0)
            s05s.append(s05)
            degs.append(degree)
        return x0s,y0s,s05s,degs

    def get_ellipse_param(self, x, y):
        x0,y0=x.mean(0),y.mean(0)
        _, s, v = np.linalg.svd(np.cov(x,y))
        s05 = s**0.5
        degree = Util.get_angle_from_v(v)
        return x0, y0, s05, degree

    def get_ellipse_fn(self, data, c=None, ratio=0.95, legend=1):
        x0s,y0s,s05s,degrees = self.get_ellipse_params(data)
        chi2_val = chi2.ppf(ratio, 2)
        co = 2 * chi2_val**0.5
        if c is None: c = "r"
        def add_ellipse(i, j, ax, handles):
            x0, y0, s05, degree = x0s[i], y0s[i], s05s[i], degrees[i]
            e = Ellipse(xy=(0,0),width=co*s05[0], height=co*s05[1], facecolor="none",edgecolor=c,)
            transf = transforms.Affine2D().rotate_deg(degree).translate(x0,y0) + ax.transData        
            e.set_transform(transf)
            ax.add_patch(e)
            if legend:
                handles.append(Ellipse(xy=(0,0),width=2, height=1, facecolor="none",edgecolor=c,label=f"Chi2_{100*ratio:.0f}%"))
            return handles
        return add_ellipse
    # def plot_pred_fn(self, data, R, color=None):
    #     def fn(i, j , ax):
    #         ax.scatter(data[:,i], data[:,j],s=1, c='k')
    #         for R in self.Rnms:
    #             # p_pred = self.

#plot box ---------------------------------------------------------------------------------

    def plot_pred_fn_R0_R1(self, R0, R1):
        data = self.p_preds[R0][R1]
        name = f'{self.dR[R1]}_Pred ({100* self.dCT[R0][R1]:.1f}%)' 
        pRange, pMin, pMax, = self.pRanges[R0], self.pMins[R0], self.pMaxs[R0]
        def fn(i, j , ax, handles=[]):
            ax.scatter(data[:,i], data[:,j],s=1, c=self.dRC[R1])
            handles.append(Line2D([0], [0], marker='o',color='w', label=name, markerfacecolor=self.dRC[R1], markersize=10))
            return handles
        return fn
            
    def box_fn_R0(self, R0, n_box=None, c="k"):
        if c is None: c = self.dRC[R0]
        pRange, pMin, pMax, = self.pRanges[R0], self.pMins[R0], self.pMaxs[R0]
        def fn(i, j , ax, handles=[]):
            if n_box is not None:
                ax.set_xlim(pMin[i]-n_box*pRange[i], pMax[i]+n_box*pRange[i])
                ax.set_ylim(pMin[j]-n_box*pRange[j], pMax[j]+n_box*pRange[j])
            ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor=c,lw=2, facecolor="none"))
            handles.append(Patch(facecolor='none', edgecolor=c, label=f"{self.dR[R0]}-Box")) 
            return handles
        return fn

    def box_fn_R0_R1(self, R0, R1, n_box=None):
        if R0!=R1:
            box_R0 = self.box_fn_R0(R0, n_box=n_box, c=None)
            box_R1 = self.box_fn_R0(R1, c="k")
            return [box_R0, box_R1]
        else:
            return [self.box_fn_R0(R0, n_box=n_box, c="k")]


    def plot_box_R0_R1(self, R0, R1, fns=[],  data=None, n_box=2, ylbl=1,  axs=None):
        if axs is None: axs = plt.subplots(1, self.npdx,  figsize=(5*self.npdx, 4), facecolor="w")[1]
        box_fns =self.box_fn_R0_R1(R0,R1, n_box=n_box)
        fns = fns + box_fns
        if data is not None: 
            fns = fns +  [self.scatter_fn(data, c=self.dRC[R1])]
        for i, ax in enumerate(axs):
            j = 0 if i == self.npdx-1 else i + 1
            handles, labels = ax.get_legend_handles_labels()
            handles = []
            for fn in fns:
                handles = fn(i, j, ax, handles)
    
            ax.legend(handles = handles)
            ax.set_xlabel(self.Pnms[self.pdx[i]])            
            # ax.annotate(f"{self.dR[R0]}-NN", xy=(0.5,0.8), xycoords="axes fraction",fontsize=15, c=self.dRC[R0])           
            # if Ps is not None: ax.set_title(f"[M/H] = {Ps[0]:.2f}, Teff={int(Ps[1])}K, logg={Ps[2]:.2f}")
            if ylbl: ax.set_ylabel(self.Pnms[self.pdx[j]])

    def plot_pred_box_R0(self, R0,  n_box=2,  axs=None, large=0):
        Rs = self.p_preds[R0].keys()
        nR = len(Rs)
        if axs is None: 
            if large:
                f, axss = plt.subplots(nR, self.npdx, figsize=(16, 4*nR), sharey="col", sharex="col", facecolor="w")
            else:
                f, axss = plt.subplots(self.npdx, nR, figsize=(20, 4*self.npdx), sharey="row", sharex="row", facecolor="w")
                axss = axss.T

        for nn, R1 in enumerate(Rs):
            fn = self.plot_pred_fn_R0_R1(R0, R1)
            self.plot_box_R0_R1(R0, R1, [fn], n_box=n_box, ylbl=0, axs=axss[nn])


    def plot_box_R0_R1_v0(self, R0, R1, data=None, Ps=None, SN=None,  n_box=2, ylbl=1,  axs=None, color=None):
        if axs is None: axs = plt.subplots(1, self.npdx,  figsize=(16, 4), facecolor="w")[1]
        pRange, pMin, pMax, = self.pRanges[R0], self.pMins[R0], self.pMaxs[R0]
        if data is None: 
            data = self.p_preds[R0][R1] if SN is None else self.ns_preds[R0][R1]
            name =  f'{self.dR[R1]}_Pred ({100* self.dCT[R0][R1]:.1f}%)' if SN is None else f'{self.dR[R1]}_SN={SN:.2f}'
        else:
            name = f'{self.dR[R0]}-NN'
        if color is None: color =self.dRC[R1]
        for i, ax in enumerate(axs):
            j = 0 if i == 2 else i + 1
            self.plot_para(data, i, j, color, ax)
            
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

    # def plot_box_R0(self, R0, data=None, Ps=None, SN=None, n_box=2,  axs=None, large=0):
    #     if axs is None: 
    #         if large:
    #             f, axss = plt.subplots(self.nR,self.npdx, figsize=(16, 4*self.nR), sharey="col", sharex="col", facecolor="w")
    #         else:
    #             f, axss = plt.subplots(self.npdx, self.nR, figsize=(20, 4*self.npdx), sharey="row", sharex="row", facecolor="w")
    #             axss = axss.T
                
    #     for i, axs in enumerate(axss):
    #         R1 = self.Rnms[i]
    #         self.plot_box_R0_R1(R0, R1, data=data, Ps=Ps, SN=SN, n_box=n_box, axs=axs, ylbl = (i==0))
    #     # plt.tight_layout()

    def plot_pred(self, R0, snrList=[], c="k",s=1, fsize=4, c1="r"):
        n_snr = len(snrList) + 1
        f, axss = plt.subplots(n_snr, self.npdx,figsize=(self.npdx* fsize, n_snr*fsize), facecolor="w")
        SN = self.s_tests[R0]
        for ii, axs in enumerate(axss.T):
            p = self.Pnms[self.pdx[ii]]
            x = self.p_tests[R0][:,ii]
            y = self.p_preds[R0][R0][:,ii]
            if n_snr==1: axs = [axs]
            axs[0].scatter(x,y,c=c,s=s, label=f"<SNR>={SN.mean():.0f}") #, label=f"{self.Pnms[pdx]}"
            RMS = np.sqrt(np.mean((x-y)**2)).round(2)
            axs[0].annotate(f"{self.dR[R0]}-NN\n{p}\n$\Delta${p}={RMS}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=fsize*3, c=c1)
            for jj, snr in enumerate(snrList):
                lb, ub = snr -5, snr + 5
                idx = np.where((SN > lb) & (SN < ub))[0]   
                xx, yy = x[idx], y[idx]
                RMS = np.sqrt(np.mean((xx-yy)**2)).round(2)
                axs[jj+1].scatter(xx, yy, s=s, c=c, label=f"{lb} < SNR < {ub}")
                axs[jj+1].annotate(f"{self.dR[R0]}-NN\n{p}\n$\Delta${p} ={RMS:.2f}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=fsize*3,c=c1)
                # axs[jj+1].legend()

            # if R is not None:
                # axs[0][ii].scatter(self.YOs[R][:,ii], self.YPs[R][:,ii], c="b",s=1, label=f"{R}")

            for ax in axs:
                ax.plot([self.pMins[R0][ii], self.pMaxs[R0][ii]], [self.pMins[R0][ii], self.pMaxs[R0][ii]], c=c1, lw=2)
                # axs[1][ii].plot(np.array([[self.pMin[ii],self.pMin[ii]], [self.pMax[ii]],self.pMax[ii]]), c="r")
                ax.set_xlim(self.pMins[R0][ii], self.pMaxs[R0][ii])
                ax.set_ylim(self.pMins[R0][ii], self.pMaxs[R0][ii])


                ax.legend(loc=2)
            # axs[1][ii].scatter(self.y_test_org[:,ii], self.dy_pred[:,ii], c="k",s=1, label=f"{self.R}")
            # if R is not None:
                # axs[1][ii].scatter(self.YOs[R][:,ii], self.dYPs[R][:,ii], c="b",s=1, label=f"{R}")

            # axs[1][ii].annotate(f"{self.R}-NN\n{c.Pnms[ii]}", xy=(0.6,0.2), xycoords="axes fraction", fontsize=20)
            # axs[1][ii].axhline(0, c='r')
            # axs[1][ii].legend()
        axs[0].set_ylabel(f"pred")
        # axs[1][0].set_ylabel(f"$\Delta$pred")

#overlap --------------------------------------------------
    # def get_overlap_R0_R1(self, R0, R1):
    #     p_pred = self.p_preds[R0][R1]
    #     mask = True
    #     for pdx in range(self.npdx):
    #         mask = mask & (p_pred[:,pdx] >= self.pMins[R0][pdx]) & (p_pred[:,pdx] <= self.pMaxs[R0][pdx])
    #     overlap = mask.sum() / self.N_test
    #     return overlap

    def get_overlap_R0_R1(self, R0, R1, pdx=None):
        if pdx is None: pdx = self.pdx
        p_pred = self.p_preds[R0][R1]
        mask = True
        for pdx_i in range(len(pdx)):
            mask = mask & (p_pred[:,pdx_i] >= self.pMins[R0][pdx_i]) & (p_pred[:,pdx_i] <= self.pMaxs[R0][pdx_i])
        overlap = mask.sum() / self.N_test
        return overlap

    def get_overlap_R0(self, R0):
        overlaps = {}
        for R1 in self.p_preds[R0].keys():
            overlaps[R1] = self.get_overlap_R0_R1(R0, R1)
        return overlaps

    def get_overlap(self):
        overlaps = {}
        for R0 in self.p_preds.keys():
            overlaps[R0] = self.get_overlap_R0(R0)
        return overlaps

    def get_overlap_mat(self, plot=1):
        Rs = self.p_preds.keys()
        nR = len(Rs)
        CT = np.zeros((nR,nR))
        for ii, R0 in enumerate(Rs):
            for jj, R1 in enumerate(Rs):
                CT[ii][jj] = self.get_overlap_R0_R1(R0, R1)
        self.CT = CT
        if plot: self.plot_heatmaps(Rs)



    def plot_heatmap(self, Rs=None, ax=None):
        if ax is None:
            f, ax = plt.subplots(figsize=(6,5), facecolor="gray")
        sns.heatmap(self.CT, vmax=1.0, ax=ax, annot=True, cmap="inferno")
        RR = [self.dR[R] for R in Rs]
        ax.set_xticklabels(RR)
        ax.set_yticklabels(RR)
        ax.set_title("overlap Heatmap")

    def plot_heatmaps(self, Rs=None):
        plt.style.use('seaborn-darkgrid')
        f, axs= plt.subplots(1,2,figsize=(12,5), facecolor="w", gridspec_kw={'width_ratios': [6, 5]})
        self.plot_heatmap(Rs=Rs, ax=axs[0])
        self.plot_heatmap_v1(Rs=Rs, ax=axs[1])

    def plot_heatmap_v1(self, Rs=None, size=2000, cut=0.005, ax=None):
        if Rs is None: 
            RRs = self.RRnms
        else:
            RRs = [self.dR[R] for R in Rs]
        nR = len(RRs)
        xv, yv = np.meshgrid(np.arange(nR), np.arange(nR))
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
        
        xlbl = [RRs[0], *RRs]
        ax.set_xticklabels(xlbl)
        ylbl = [RRs[0], *RRs[::-1]]
        ax.set_yticklabels(ylbl, rotation=90,  verticalalignment='center', horizontalalignment='right')
        ax.set_title(f"Error > {100*cut}%")

    def plot_snr_hist(self, snr, mag=19, exp_count=20, exp_time=900, ax=None):
        if ax is None:
            f, ax = plt.subplots(figsize=(5,4), facecolor="w")
        _=ax.hist(snr, bins=100, density=1, label=f"mag={mag}", color="gray")
        ax.set_xlabel("SNR")
        ax.set_ylabel("density of spec")
        ax.set_title(f"exp_count = {exp_count}, exp_time = {exp_time}s")
        ax.legend()


#DNN--------------------------------------------------

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

    def trans_predict_W(self, data, W, R0, dnn=None):
        pc_data = self.transform_W_R(data, W, R0)
        p_pred = self.predict(pc_data, R0, dnn=None)
        return p_pred
    
    def trans_predict(self, data, R0, dnn=None):
        pc_data = self.transform_W_R(data, R0)
        p_pred = self.predict(pc_data, R0, dnn=None)
        return p_pred
    



#else----------------------------------------------

    # def load_snr_flux_idx(self, W, R0, SNR_PATH=None, idx=0):
    #     RR = self.dR[R0]
    #     if SNR_PATH is None: SNR_PATH=f"{self.dataDir}/{RR}/snr/{W}_{idx}.h5"    
    #     with h5py.File(SNR_PATH, "r") as f:
    #         wave = f["wave"][:]
    #         flux = f["flux"][:]
    #         para = f["para"][:]
    #     dSnr={}
    #     with h5py.File(SNR_PATH, "r") as f:
    #         for snr in self.snrList:
    #             dSnr[snr] = f[f"snr_{snr}"][()]
    #     if self.pdx is not None: para = para[self.pdx]
    #     return wave, flux, para, dSnr


    # def predict_snr_flux_R0_i(self, i, R0=None, W="RedM", N=100, step=20):
    #     wave, flux, para, dSnr = self.load_snr_flux(W, R0, idx=i)
    #     fluxL = self.Util.resampleFlux_i(flux, step=step)
    #     w = self.dWw[W][1]
    #     dSN_preds_i = {}
    #     for snr, err in dSnr.items():
    #         fluxNL = self.add_noise_N(fluxL, err, N=N, step=step)
    #         lognorm_fluxL = self.Util.lognorm_flux(fluxNL)
    #         dSN_preds_i[snr] = self.trans_predict(lognorm_fluxL, W=w, R0=R0)
    #     return dSN_preds_i, para

    # def predict_snr_flux_R0(self, R0, W="RedM", nSN=10):
    #     paras = []
    #     dSN_preds = {}
    #     for i in nSN:
    #         dSN_preds[i], para = self.predict_snr_flux_R0_i(i, R0=R0, W=W)
    #         paras.append(para)
    #         paras=np.vstack(paras)
    #     return dSN_preds, paras


    # def prepare_snr_flux_i(self, i, R0=None, W="RedM", N=100, step=20):
    #     wave, flux, para, dSnr = self.load_snr_flux(W, R0, idx=i)
    #     fluxL = self.Util.resampleFlux_i(flux, step=step)
    #     dSN_flux_i = {}
    #     for snr, err in dSnr.items():
    #         fluxNL = self.add_noise_N(fluxL, err, N=N, step=step)
    #         lognorm_fluxL = self.Util.lognorm_flux(fluxNL)
    #         dSN_flux_i[snr] = lognorm_fluxL
    #     return dSN_flux_i, para
    