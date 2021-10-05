import time
# import fbpca
import logging
import numpy as np
import cupy as cp
import h5py
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lv.pcp.pcpc import pcp_cupy
from lv.base.baseLL import KLine
from cuml import UMAP



class DataLoader(object):
    def __init__(self):
        ################################ Flux Wave ###############################
        self.Ws = {"Blue": [3800, 6500, 2300, "Blue"], "RedL": [6300, 9700, 3000, "RedL"], "RedM": [7100, 8850, 5000, "RedM"],
                   "NIR": [9400, 12600, 4300, "NIR"], "RMLL": [7100, 8850, 500, "RMLL"]}

        self.Rs = { "M": [[-2.5, 0.0], [3500, 5000], [0.0, 2.0],[-0.75, 0.5], [-0.25, 0.5]], 
                    "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]],
                    "C": [[-2.0, 0.0], [4500, 6000], [4.0, 5.0],[-0.75, 0.5], [-0.25, 0.5]], 
                    "B": [[-2.5,-1.5], [7000, 9500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]],
                    "R": [[-1.0, 0.0], [5000, 6500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]], 
                    "G": [[-2.5,-1.0], [3500, 5500], [0.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]]}
        self.W = None
        self.R = None
        self.nw = None
        self.v = None
        self.nv = None
        self.RNms = {"M": "M31G", "W": "MWW", "C": "MWC", "B": "BHB", "R": "RHB", "G":"DGG"}
        self.RNm = None
        self.PNms = ["[M/H]", "Teff", "Logg", "[C/M]", "[a/M]"]
        self.Fs = {}
        
        self.FNs = {}
        self.flux = None
        self.wave = None
        self.para = None

        self.mask = None
        self.nmask = None
        self.pmax = None

        self.M = None
        self.N = None
        self.Mv = None
        self.Nv = None
        self.Mw = None
        self.Nw = None
        self.MLv = None
        self.NLv = None
        self.MSv = None
        self.NSv = None
        self.Xv = None
        self.Xname = None
        self.pcpM = None
        self.pcpN = None
        self.pcpFlux = None
        self.npcpFlux = None

        self.pcaFlux = None
        self.npcaFlux = None

        self.nXv = None
        self.Fs =  {"M": {}, "N": {}}
        self.Xdx = {"M": {}, "N": {}}
        self.XdxAll = None
        self.Cdx = {}
        self.CdxAll = None
        self.npcpFlux = None
        self.cmap="YlGnBu"
        self.color = {"T": "gist_rainbow", "L": "turbo", "F": "plasma", "C": "gist_rainbow", "O":"winter"}
        self.ps =  [["p0","p1", "p2", "p3", "p4"],["p5","p6", "p7", "p8", "p9"],["p10","p11", "p12", "p13", "p14"],["p15","p16", "p17", "p18", "p19"]]
        self.name = None
        self.lick = None
        self.l = None
        self.AL = None

################################ Flux Wave #####################################
    def prepare_lines(self):
        self.l=KLine(self.W[3])
        self.AL = pd.read_csv("../data/Klines.csv")

################################ Flux Wave #####################################
    def prepare_data_custom(self, W, flux, wave, para, lbl=None):
        self.W = self.Ws[W]
        self.nwave = wave
        self.nw = len(wave)
        self.RNm = "ALL"
        self.dfpara = pd.DataFrame(data=para, columns=["F","T","L","C","O"])
        if lbl is not None:
            self.lbl = lbl
        print(flux.shape, wave.shape, para.shape)

        flux = cp.asarray(flux, dtype=cp.float32)
        wave = cp.asarray(wave, dtype=cp.float32)
        self.flux = cp.clip(-flux, 0.0, None)
        self.wave = wave
        


    
    def prepare_data(self, W, R, flux, wave, para, fix_CO=False):
        self.W = self.Ws[W]
        self.R = self.Rs[R]
        self.RNm = self.RNms[R] 
        self.nwave = wave        
        index = self.get_flux_in_Prange(para, fix_CO=fix_CO)
        flux  = flux[index]
        # flux, wave = self.get_flux_in_Wrange(flux, wave)
        print(f"flux: {flux.shape[0]}, wave: {len(self.nwave)}")
        self.name = f"{self.RNms[R]} in {W}"

        #gpu only
        flux = cp.asarray(flux, dtype=cp.float32)
        wave = cp.asarray(wave, dtype=cp.float32)
        self.flux = cp.clip(-flux, 0.0, None)
        self.wave = wave
        self.nw = len(wave)

    def get_LL(self):            
        df = pd.read_csv(f"/scratch/ceph/szalay/swei20/LL/kurucz/gfall_vac_{self.W[3]}.csv")
        return df
    
    def get_flux_in_Prange(self, para, fix_CO=True):
        Fs, Ts, Ls,_,_ = self.R
        dfpara = self.init_para(para)
        if fix_CO:
            dfpara = dfpara[(dfpara["O"] == 0.0)]
            # dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["O"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        maskF = (dfpara["F"] >= Fs[0]) & (dfpara["F"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["L"] >= Ls[0]) & (dfpara["L"] <= Ls[1]) 
        mask = maskF & maskT & maskL
        self.dfpara = dfpara[mask]
        self.para = np.array(self.dfpara.values, dtype=np.float16)
        return self.dfpara.index

    def get_flux_in_Wrange(self, flux, wave):
        Ws = self.W
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]

    def init_para(self, para):
        return pd.DataFrame(data=para, columns=["F","T","L","C","O"])
####################################### SVD #######################################
    def _svd(self, X):
        return cp.linalg.svd(X, full_matrices=0)

    def get_eigv(self, X, top=5, out_w=False):
        _,w,v = self._svd(X)
        if out_w: return v[:top], w
        return v[:top] 

    def plot_eigv(self, v, top=5, step=0.3, isM=None, name=None, ax=None):
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        nv = cp.asnumpy(v[:top])
        for i in range(min(len(nv),top)):
            ax.plot(self.nwave, nv[i] + step*(i+1))
        ax.set_ylabel(f"Top {top} {name} {self.name}")
        self.get_wave_axis(ax=ax)
        if isM is not None: self.plot_mask_below(isM=isM, large=1, ax=ax) 
        

    def plot_V(self, nv, top=5, step=0.3, ax=None):
        wave = self.nwave        
        size = top // 5
        if ax is None: ax = plt.subplots(1, figsize=(16,3 * size),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(wave, nv[i] + step*(i+1))
        self.get_wave_axis(ax=ax)

    def plot_IV(self, sdx, top=5, step=0.3, ax=None):
        size = top // 5
        if ax is None: ax = plt.subplots(1, figsize=(16,4 * size),facecolor="w")[1]
        nv = np.abs(self.nXv[sdx])
        nm = [self.Xname[s] for s in sdx]
        for i in range(min(len(nv),top)):
            ax.plot(self.nwave, nv[i] + step*(i+1), c="k") #lw=2, label=nm[i])
        # self.get_wave_axis(ax=ax, xgrid=0)


    def plot_v(self, vs, idx, nidx=None, c=None, ax=None):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,5))[1]
        vs = cp.asnumpy(vs)
        v = vs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)
    
    def plot_nvs(self, nvs, idxs, nidxs=None, c="k", ax=None, fineW=0):
        if nidxs is None: nidxs = self.Xname
        n = len(idxs)
        f, axs = plt.subplots(n,1, figsize=(16,2*n))
        for i in range(n):
            ax = axs[i]
            nv = nvs[idxs[i]]
            nidx = nidxs[idxs[i]]
            ax.plot(self.nwave, nv, label=nidx, c=c)
            if not fineW: 
                self.get_wave_axis(ax=ax)
            else:
                ax.xaxis.grid(1)
            ax.legend(loc=1)


    def plot_nv(self, nvs, idx, nidx=None, c="k", ax=None, fineW=0, fs=1):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,fs))[1]
        v = nvs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)
        if not fineW: 
            self.get_wave_axis(ax=ax)
        else:
            ax.xaxis.grid(1)

    def init_pcp(self, step=0.3):
        self.v = self.get_eigv(self.flux, top=200, out_w=False)
        self.nv = cp.asnumpy(self.v)
        self.plot_V(self.nv, step=step)
        plt.ylabel(self.name)
####################################### Mask #######################################
    def plot_mask(self, mask, ymin=0, ymax=0.1, c='r', lw=1, ax=None, fineW=0):
        ax = ax or plt.subplots(figsize=(16,1))[1]
        ax.vlines(self.nwave[mask], ymin=ymin, ymax=ymax, color=c, lw=lw)
        if not fineW: self.get_wave_axis(ax=ax)



    def plot_mask_below(self, isM=1, large=0, ax=None):
        ymin = -self.pmax ** 0.5 if large else -self.pmax*0.1
        if isM:
            ax.vlines(self.nwave[self.nmask], ymin=ymin, ymax=0.0, color="r", alpha=0.5)
        else:
            ax.vlines(self.nwave[~self.nmask], ymin=ymin, ymax=0.0, color="g", alpha=0.5)


    def get_mask_from_nv(self, nv, k=5, q=0.8):
        vv = np.sum(nv[:k]**2, axis=0)
        cut = np.quantile(vv, q)
        mask = vv > cut        
        return mask, vv

    def get_peaks(self, nv=None, k=100, q=0.6, prom=0.2):
        if nv is None: nv = self.nv
        mask, nvv = self.get_mask_from_nv(nv, k=k, q=q)   
        nvv[~mask]  = 0.0
        peaks, prop = self.find_peak(nvv, prom)
        return peaks, prop, nvv

    def find_peak(self, nv, prom):
        peaks, prop = find_peaks(nv, prominence=(prom, None))
        return peaks, prop

    def plot_rfPC(self, pdx=0,top=10, rng=None):
        f, axs = plt.subplots(top,1, figsize=(16,2*top), facecolor="w")
        if top ==1: axs=[axs]
        for vdx in range(top):
            ax=axs[vdx]
            self.get_wave_axis(wave=rng, ax=ax)

            PC = self.nXv[self.Fs[pdx][vdx]]
            PCN = self.Xname[self.Fs[pdx][vdx]]
            self.plot_rfPC_v(PC,PCN, ax=ax)

    def plot_rfPCN(self, pdx=0,top=10, rng=None):
        f, axs = plt.subplots(top,1, figsize=(16,2*top), facecolor="w")
        if top ==1: axs=[axs]
        for vdx in range(top):
            ax=axs[vdx]
            self.get_wave_axis(wave=rng, ax=ax)

            PC = self.nXv[40 + self.FNs[pdx][vdx]]
            PCN = self.Xname[40 + self.FNs[pdx][vdx]]
            self.plot_rfPC_v(PC,PCN, ax=ax)

    def plot_rfPC_v(self, PC, PCN, prom=0.1, ax=None):
        peaks, prop = self.find_peak(abs(PC), prom)
        self.plot_peak_Z(PC,PCN, peaks,prop, ax=ax )

    def plot_peak_Z(self, nv, nv_name, peaks, prop, ax=None):
        if ax is None: ax = plt.subplots(figsize=(16,1))[1]
        ax.plot(self.nwave, abs(nv), c="k", label=nv_name)
        for (pval,Y,W,ZN) in self.get_peaks_Z(peaks,prop):
            self.plot_peak_from_PYWZ(pval,Y,W,ZN, ax=ax)
        ax.legend()


    def plot_peak_from_PYWZ(self, pval,Y,W,ZN, ax=None):
        if ax is None: ax = plt.subplots(figsize=(16,1))[1]
        # ax.plot(pval, Y,"bx", markersize=10)
        ax.vlines(W, ymin=1.1*Y, ymax=3.*Y, color="r")
        ax.annotate(f"{ZN}", (W+0.1, 2.5*Y),color="r")

    def get_WZ_from_PY(self, pval, Y):
        try:
            KL = self.AL
            dfP = KL[(KL["W"] > pval-3) & (KL["W"] < pval+3)]
            assert (len(dfP) == 1)
            ZN = dfP.iloc[0]["Z"]
            W = dfP.iloc[0]["W"]
        except:
            try:
                KL = self.l.dfSL
                dfP = KL[(KL["W"] > pval-3) & (KL["W"] < pval+3)]
                assert(len(dfP) > 0)
                II= dfP["I"].mode()
                if len(II) > 1:
                    dfP = dfP[dfP["I"]==II.max()]
                else:                    
                    dfP = dfP[dfP["I"]==II.values[0]]
                W = dfP["W"].mean()
                Z = dfP["Z"].unique()
            except:
                KL = self.l.dfLL
                dfP = KL[(KL["W"] > pval-3) & (KL["W"] < pval+3)]
                assert(len(dfP) > 0)
                W,_,Z,_ = dfP.iloc[dfP["I"].argmax()].values        
            ZN = self.l.ZNms[int(Z)]
        return W, ZN

    def get_peaks_Z(self, peaks, prop):
        for ii, peak in enumerate(peaks):
            pval = self.nwave[peak]
            Y = prop["prominences"][ii]
            W, ZN = self.get_WZ_from_PY(pval, Y)
            yield pval, Y, W, ZN


    def plot_peaks(self, nvv, peaks, k, prom, ax=None):
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.plot(self.nwave, nvv, c="k", label=f"leverage score k={k}")
        vpeaks = nvv[peaks]
        ax.plot(self.nwave[peaks], vpeaks, "bx", markersize=10, label=f"prominence={prom}")
        self.get_wave_axis(ax=ax)
        ax.legend()

    def get_mask_from_peaks(self, peaks, prop):
        prom = prop["prominences"]
        lb = prop["left_bases"]
        ub = prop["right_bases"]
        mask = np.zeros_like(self.nwave, dtype=bool)
        for i in range(len(peaks)):
            mask[lb[i]:ub[i]+1] = True
        self.pmax = np.max(prom)
        self.nmask = mask
        self.mask = cp.asarray(mask, dtype=cp.bool)

    def plot_mask_from_peaks(self, peaks, prop, nvv, ax=None):
        self.get_mask_from_peaks(peaks, prop)
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.plot(self.nwave, nvv, c="k")
        ax.plot(self.nwave[peaks], nvv[peaks], "bx", markersize=10)
        self.plot_mask_below(isM=1, ax=ax)
        self.get_wave_axis(ax=ax)

    def get_wave_axis(self, wave= None, ax=None, xgrid=True):
        if wave is None: 
            ax.set_xlim(self.nwave[0]-1, self.nwave[-1]+2)
            ax.set_xticks(np.arange(self.W[0], self.W[1], 200))  
        else:
            ax.set_xlim(wave[0]-1, wave[-1]+2)
            n = (wave[-1]-wave[0]) // 10
            ax.set_xticks(np.arange(int(wave[0]), np.ceil(wave[-1]), n))
        ax.xaxis.grid(xgrid)

    def plot_MN_mask(self, idx=0, ax=None):
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.vlines(self.nwave[self.nmask], ymin=0.0, ymax=self.pmax ** 0.5, color="r", alpha=0.3)
        ax.vlines(self.nwave[~self.nmask], ymin=-self.pmax ** 0.5, ymax=0.0, color="g", alpha=0.3)
        ax.plot(self.nwave, self.nv[idx], c="k")        
        self.get_wave_axis(ax=ax, xgrid=0)

####################################### M N #######################################
    def get_MN(self, mask, top=5):
        self.M = self.flux[:,  mask]
        self.N = self.flux[:, ~mask]    
        Mv, self.Mw =self.get_eigv(self.M, top=top, out_w=True)
        Nv, self.Nw =self.get_eigv(self.N, top=top, out_w=True)
        Mv = self.get_xv(Mv, isM=1)
        Nv = self.get_xv(Nv, isM=0)
        self.nMv = cp.asnumpy(Mv)
        self.nNv = cp.asnumpy(Nv)

    def get_xv(self, v, isM=1):
        xv = cp.zeros((v.shape[0], self.nw))
        if isM:
            xv[:, self.mask] = v
        else:
            xv[:, ~self.mask] = v
        return xv

    def plot_MN(self, step=0.3, axs=None):
        if axs is None: axs = plt.subplots(2,1,figsize=(16,10))[1]
        self.plot_V(self.nMv, top=5, step=step, ax=axs[0])
        self.plot_mask_below(isM=1, large=1, ax=axs[0])

        self.plot_V(self.nNv, top=5, step=step, ax=axs[1])
        self.plot_mask_below(isM=0, large=1, ax=axs[1])

####################################### PCP #######################################
    def _pcp(self, X, delta=1e-6, mu=None, lam=None, norm=None, maxiter=50):
        XL, XS, (_,_,XLv) = pcp_cupy(X, delta=delta, mu=mu, lam=lam, norm=norm, maxiter=maxiter)
        XSv = self.get_eigv(XS, top = 30)
        print(f"L{XLv.shape}, S{XSv.shape}")
        return XL, XS, XLv, XSv

    def eval_pcp(self, XLv, XSv, isM=1, step=0.3, ax=None):
        if ax is None: ax = plt.subplots(2, 1, figsize=(16,6),facecolor="w")[1]
        XLv = self.get_xv(XLv, isM=isM)
        XSv = self.get_xv(XSv, isM=isM)
        nXLv = cp.asnumpy(XLv)
        print(nXLv.shape)
        self.plot_V(nXLv, top=5, step=step, ax=ax[0])
        nXSv = cp.asnumpy(XSv)
        self.plot_V(nXSv, top=5, step=step, ax=ax[1])
        return nXLv, nXSv

    def pcp_transform(self, MLv, MSv, NLv, NSv, top=20):
        MLv, MSv, NLv, NSv = MLv[:top], MSv[:top], NLv[:top], NSv[:top] 
        Mvs = cp.vstack((MLv, MSv))
        self.pcpM = cp.dot(self.M, Mvs.T)
        Nvs = cp.vstack((NLv, NSv))
        self.pcpN = cp.dot(self.N, Nvs.T)
        self.pcpFlux = cp.hstack((self.pcpM, self.pcpN))
        self.MLv = self.get_xv(MLv, isM=1)
        self.MSv = self.get_xv(MSv, isM=1)
        self.NLv = self.get_xv(NLv, isM=0)
        self.NSv = self.get_xv(NSv, isM=0)
        self.Xv = cp.vstack((self.MLv, self.MSv, self.NLv, self.NSv))
        self.Xname = self.get_Xname(top)
        self.nPC = top
        self.nPC2 = top*2


    def get_Xname(self, top=15):
        name = []
        for X in ["M", "N"]:
            for XX in ["L", "S"]:
                name += [f"{X}{XX}{i}" for i in range(top)]
        return name

    def pcp_np(self, PATH=None, save=0):
        self.nXv = cp.asnumpy(self.Xv)
        self.npcpFlux = cp.asnumpy(self.pcpFlux)
        if save:
            if PATH is None: PATH= f"/scratch/ceph/swei20/data/dnn/{self.RNm}/bosz_pcp.h5"
            print(PATH)
            self.save_dnn_pcp(PATH)

    def save_dnn_pcp(self, DNN_PCP_PATH):
        ww = self.W[3][:1]
        with h5py.File(DNN_PCP_PATH, 'a') as f:
            f.create_dataset(f"flux{ww}", data=self.npcpFlux, shape=self.npcpFlux.shape)
            f.create_dataset(f"pcp{ww}", data=self.nXv, shape=self.nXv.shape)
            f.create_dataset(f"para{ww}", data=self.dfpara.values, shape=self.dfpara.shape)
            f.create_dataset(f"pc{ww}", data=self.nv, shape=self.nv.shape)

    def save_dnn_rf(self, PATH=None):
        if PATH is None: PATH= f"/scratch/ceph/swei20/data/dnn/{self.RNm}/bosz_pcp.h5"
        ww = self.W[3][:1]
        with h5py.File(PATH, 'a') as f:
            f.create_dataset(f"Xdx{ww}", data=self.XdxAll, shape=self.XdxAll.shape)



    def pcp_ntransform(self, MLv, MSv, NLv, NSv, out=0):
        self.npcp20 = np.vstack([MLv[:5],MSv,NLv[:5],NSv])
        flux = cp.asnumpy(self.flux)
        nflux20 = np.dot(flux, self.pcp20.T)
        for i in range(self.nPC):
            self.dfpara[f"p{i}"] = nflux20[:,i]
        if out:
            return nflux20, npcp20

    def save_PCP(self, PCP20_PATH, nflux20, npcp20):
        with h5py.File(PCP20_PATH, 'w') as f:
            f.create_dataset("flux", data=nflux20, shape=nflux20.shape)
            f.create_dataset("pcp", data=npcp20.values, shape=npcp20.shape)
            f.create_dataset("wave", data=nwave, shape=nwave.shape)
            f.create_dataset("para", data=self.dfpara.values, shape=self.dfpara.shape)


    # def cluster_X(self, top=20, plot=1, X=None, p_num=5):
    #     if plot:
    #         f, axs = plt.subplots(p_num, 1, figsize=(16,2*p_num),facecolor="w")
    #     for pdx in range(p_num):
    #         ax = axs[pdx] if plot else None
    #         self.get_Xrf(pdx=pdx, top=20, plot=plot, ax=ax, X=X)
    #     sdx=set()
    #     stop=0
    #     for i in range(self.nPC2): # ML + MS
    #         if stop==0:
    #             for j in range(p_num):
    #                 if stop==0:
    #                     sdx.add(self.Fs[X][j][i])
    #                     if len(sdx) > (top-1): stop=1
    #     self.Xdx[X] = list(sdx)
    #     self.XdxAll = np.array(self.Xdx["M"] + [self.nPC2 + ii for ii in self.Xdx["N"]])
    #     print(sdx)

    def get_all_Xrf(self, top=20, plot=1, X=None, p_num=5):
        if plot:
            f, axs = plt.subplots(p_num, 1, figsize=(16,2*p_num),facecolor="w")
        for pdx in range(p_num):
            ax = axs[pdx] if plot else None
            self.get_Xrf(pdx=pdx, top=20, plot=plot, ax=ax, X=X)
        sdx=set()
        stop=0
        for i in range(self.nPC2): # ML + MS
            if stop==0:
                for j in range(p_num):
                    if stop==0:
                        sdx.add(self.Fs[X][j][i])
                        if len(sdx) > (top-1): stop=1
        self.Xdx[X] = list(sdx)
        print(sdx)

    def plot_XdxAll(self, pcp=1, top=6, rng=None, rfr=1):
        if pcp:
            if rfr: 
                sdx = np.array(self.Xdx["M"] + [self.nPC2 + ii for ii in self.Xdx["N"]])
                self.XdxAll = sdx            
            else:
                sdx = np.append(self.Cdx["M"][:top],  self.nPC2 + self.Cdx["N"][:top])
                self.CdxAll = sdx
            V = self.nXv
            N = self.Xname
        else:
            sdx = self.Cadx
            V = self.nv
            N = [f"v{i}" for i in range(len(sdx))]

        f, axs = plt.subplots(top*2,1, figsize=(16,2*top), facecolor="w")
        if top==1: axs=[axs]
        for vdx in range(top*2):
            ax=axs[vdx]
            self.get_wave_axis(wave=rng, ax=ax)
            PC = V[sdx[vdx]]
            PCN = N[sdx[vdx]]
            self.plot_rfPC_v(PC,PCN, ax=ax)
 


    def plot_Xdx(self, top=20, rng=None, X=None, rfr=1):
        sdx = self.Xdx if rfr else self.Cdx 
        offset=self.nPC2 if X =="N" else 0

        f, axs = plt.subplots(top,1, figsize=(16,2*top), facecolor="w")
        if top==1: axs=[axs]
        for vdx in range(top):
            ax=axs[vdx]
            self.get_wave_axis(wave=rng, ax=ax)
            PC = self.nXv[offset + sdx[X][vdx]]
            PCN = self.Xname[offset + sdx[X][vdx]]
            self.plot_rfPC_v(PC,PCN, ax=ax)
 
    # def get_dfumap(self,)


    def get_X_cluster(self, data=None, top=20, plot=1, ax=None, X=None):
        if data is None:
            store_dx = self.Cdx
            if X == "M": 
                data = self.npcpFlux[:,:self.nPC2]
            elif X == "N":
                data = self.npcpFlux[:,self.nPC2:]
            else:
                raise ValueError("X must be M or N")
        else:
            if self.npcaFlux is None:
                self.pcaFlux = self.flux.dot(self.v.T)
                self.npcaFlux = cp.asnumpy(self.pcaFlux)
            data = self.npcaFlux
 
        rf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=100, max_features=40)
        rf.fit(data, self.lbl)
        sdx = rf.feature_importances_.argsort()[::-1]
        if X is not None:
            self.Cdx[X] = sdx
            if plot: 
                self.plot_Xrf(rf, sdx[:top], log=0, X=X, ax=ax)
                if ax is None: ax=plt.gca()
                ax.axhline(0.1, color="r", ls="--")
        else:
            self.Cadx = sdx
            if plot:
                self.barplot_rf(rf, sdx[:top], log=0, ax=ax)

    def barplot_rf(self, rf, sdx, log=1, color="k", ax=None):
        if ax is None: ax =plt.subplots(1, figsize=(16,1), facecolor="w")[1]
        ax.bar([f"v{sdx[i]}" for i in range(len(sdx))],  rf.feature_importances_[sdx], log=log, color=color)


    def get_Xrf(self, pdx=1, fdx=None, top=20, plot=1, ax=None, X=None):
        if fdx is None: 
            if X == "M": 
                data = self.npcpFlux[:,:self.nPC2]
            elif X == "N":
                data = self.npcpFlux[:,self.nPC2:]
            else:
                raise ValueError("X must be M or N")
        else:
            data = self.npcpFlux[:,fdx]

        rf = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=100, max_features=30)

        rf.fit(data, self.para[:, pdx])
        sdx = rf.feature_importances_.argsort()[::-1]
        self.Fs[X][pdx] = sdx
        if plot: 
            self.plot_Xrf(rf, sdx[:top], log=1, X=X, ax=ax)
            if ax is None: ax=plt.gca()
            ax.annotate(f"{self.PNms[pdx]}", xy=(0.9,0.5), xycoords="axes fraction", fontsize=20)

    def plot_Xrf(self, rf, sdx, log=1, X=None, color="k", ax=None):
        if ax is None: ax =plt.subplots(1, figsize=(16,1), facecolor="w")[1]
        if   X=="M":
            ax.bar([self.Xname[sdx[i]] for i in range(len(sdx))], rf.feature_importances_[sdx], log=log, color=color)
        elif X=="N":
            ax.bar([self.Xname[self.nPC2 + sdx[i]] for i in range(len(sdx))], rf.feature_importances_[sdx], log=log,color=color)
        else:
            raise ValueError("X must be M or N")

######################################## M LS #######################################
    def p(self, idx1, idx2, para, large=0, data=None):
        if large:
            plt.figure(figsize=(8,6), facecolor="w")
            s=5
        else:
            plt.figure(figsize=(6,4), facecolor="w")
            s=3
        if data is None: data = self.dfpara
        sns.scatterplot(
            data=data,x=f"p{idx1}", y=f"p{idx2}", hue=para, marker="o", s=s, edgecolor="none",palette=self.color[para])
        plt.title(self.name)

    def prf(self, idx1, idx2, para, large=0):
        plt.scatter(self.npcpFlux[:, idx1], self.npcpFlux[:,idx2], c=self.dfpara[para], marker=".", s=5, edgecolor="none", cmap=self.color[para])

    def pp(self, idx, para, data=None):
        if data is None: data = self.dfpara
        sns.pairplot(
            data,
            x_vars=self.ps[idx],
            y_vars=self.ps[idx],
            hue=para,
            plot_kws=dict(marker="o", s=2, edgecolor="none"),
            diag_kws=dict(fill=False),
            palette=self.color[para],
            corner=True
        )

    def plot_pcp(self, M, L, S, u, s, v, cmap="hot"):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(cp.abs(M), aspect="auto", cmap=cmap,)
        plt.title("Original Flux")
        plt.subplot(2, 2, 2)
        plt.imshow(cp.abs(L), aspect="auto", cmap=cmap)
        plt.title("Low rank matrix")
        plt.subplot(2, 2, 3)
        plt.imshow(cp.abs(S), cmap=cmap, aspect="auto", )
        plt.title("Sparse matrix")
        plt.subplot(2, 2, 4)
        for i in range(min(len(v),5)):
            plt.plot(self.wave, v[i] + 0.3*(1 + i))
        plt.plot(self.nwave, cp.mean(cp.abs(S), axis=0), c="k")
        # plt.imshow(cp.dot(u, cp.dot(cp.diag(s), v)), cmap="gray")
        plt.title("L & S")
        plt.show()



    def get_lick(self):
        dBands = {}
        dBands['CN'] = [[4142,4177]]
        dBands['Ca'] = [[3899, 4003], [4222, 4235], [4452, 4475], [8484, 8513],
                        [8522, 8562], [8642, 8682], [6358, 6402], [6775, 6900]]
        dBands['Fe'] = [[4369, 4420], [4514, 4559], [4634, 4720], 
                        [4978, 5054], [5246, 5286], [5312, 5363], 
                        [5388, 5415], [5697, 5720], [5777, 5797]]
        dBands['G']  = [[4281, 4316]]
        dBands['H']  = [[4839, 4877], [4084, 4122], [4320, 4364]]
        dBands['Mg'] = [[4761, 4799], [5069, 5134], [5154, 5197]]
        dBands['Na'] = [[8164, 8229], [8180, 8200], [5877, 5909]]
        dBands['Ti'] = [[6190, 6272], [6600, 6723], [5937, 5994], 
                        [7124, 7163], [7643, 7717], [5445, 5600], [4759, 4800]]

        cBands = {}
        cBands['CN'] = 'darkblue'
        cBands['Ca'] = 'red'
        cBands['Fe'] = 'yellow'
        cBands['G']  = 'purple'
        cBands['H']  = 'cyan'
        cBands['Mg'] = 'pink'
        cBands['Na'] = 'orange'
        cBands['Ti'] = 'lime'
        self.lick = dBands
        self.lick_color = cBands

    def plot_lick(self, ax=None, fineW=1):
        ax = ax or plt.subplots(figsize=(16,2))[1]
        # ax.grid(True)
        ax.set_ylim(0, 1)
        l_max = 1
        if self.lick is None: self.get_lick()
        for idx, (key, vals) in enumerate(self.lick.items()):
            for val in vals:
                val_idx = np.digitize(val, self.nwave)
                ax.axvspan(self.nwave[val_idx[0]], self.nwave[val_idx[1]], ymin=0, ymax=l_max, color = self.lick_color[key], label = key)
    #     for idx, (key, vals) in enumerate(lines.items()):
    #         ax.vlines(vals, -l_max, 0, color = next(color), label = key, linewidth = 2)

        self.set_unique_legend(ax)
        if fineW: self.get_wave_axis(ax=ax,)
        # self.set_wv_ticks(ax, lim=True)
        # ax.set_ylabel('LICK')
    def plot_O(self, x, c="r",ls=":", ax=None):
        ax = ax or plt.gca()
        if x // 1000 == 8:
            ax.axvline(8448.57, c=c, linestyle=ls, label="O I")
        elif x // 1000 == 7:
            ax.axvline(7777.53, c=c, linestyle=ls, label="O I")
        ax.legend()



    def set_unique_legend(self, ax, fix_ncol=1, loc="upper right"):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        vals = by_label.values()
        if not fix_ncol:
            nn = len(vals) // 2
            ax.legend(vals, by_label.keys(), ncol=nn, loc=loc)
####################################### SAVE #######################################
# PCP_PATH = '/scratch/ceph/szalay/swei20/AE/PCP_LL.h5'

def save_M(self, PCP_PATH, ML, MS, MLv, MSv):
    with h5py.File(PCP_PATH, 'w') as f:
        f.create_dataset("flux", data=cp.asnumpy(self.flux), shape=self.flux.shape)
        f.create_dataset("wave", data=cp.asnumpy(self.wave), shape=self.wave.shape)
        f.create_dataset("para", data=self.dfpara.values, shape=self.dfpara.shape) 
        f.create_dataset("ML", data=cp.asnumpy(ML), shape=ML.shape)
        f.create_dataset("MS", data=cp.asnumpy(MS), shape=MS.shape)
        f.create_dataset("MLv", data=cp.asnumpy(MLv), shape=MLv.shape) 
        f.create_dataset("MSv", data=cp.asnumpy(MSv), shape=MSv.shape) 

def save_N(self, PCP_PATH, NL, NS, NLv, NSv):
    with h5py.File(PCP_PATH, 'a') as f:
        f.create_dataset("NL", data=cp.asnumpy(NL), shape=NL.shape)
        f.create_dataset("NS", data=cp.asnumpy(NS), shape=NS.shape)
        f.create_dataset("NLv", data=cp.asnumpy(NLv), shape=NLv.shape) 
        f.create_dataset("NSv", data=cp.asnumpy(NSv), shape=NSv.shape) 

