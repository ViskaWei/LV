import time
# import fbpca
import logging
import numpy as np
import cupy as cp
import h5py
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lv.pcp.pcpc import pcp_cupy




class DataLoader(object):
    def __init__(self):
        ################################ Flux Wave ###############################
        self.Ws = {"Blue": [3800, 6500, 2300, "Blue"], "RedL": [6300, 9700, 3000, "RedL"], "RedM": [7100, 8850, 5000, "RedM"],
                   "NIR": [9400, 12600, 4300, "NIR"]}

        self.Ps = { "M": [[-2.5, 0.0], [3500, 5000], [0.0, 2.0],[-0.75, 0.5], [-0.25, 0.5]], 
                    "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]],
                    "C": [[-2.0, 0.0], [4500, 6000], [4.0, 5.0],[-0.75, 0.5], [-0.25, 0.5]], 
                    "B": [[-2.5,-1.5], [7000, 9500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]],
                    "R": [[-1.0, 0.0], [5000, 6500], [2.0, 3.0],[-0.75, 0.5], [-0.25, 0.5]], 
                    "G": [[-2.5,-1.0], [3500, 5500], [0.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]]}
        self.W = None
        self.nw = None
        self.P = None
        self.v = None
        self.nv = None
        self.Names = {"M": "M31 Giant", "W": "MW Warm", "C": "MW Cool", "B": "BHB", "R": "RHB", "G":"DwarfG Giant"}
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

        self.nXv = None
        self.npcpFlux = None
        self.pnames = ["F","T","L","C","O"]



        self.cmap="YlGnBu"
        self.color = {"T": "gist_rainbow", "L": "turbo", "F": "plasma", "C": "gist_rainbow", "O":"winter"}
        self.ps =  [["p0","p1", "p2", "p3", "p4"],["p5","p6", "p7", "p8", "p9"],["p10","p11", "p12", "p13", "p14"],["p15","p16", "p17", "p18", "p19"]]
        self.name = None
        self.lick = None

################################ Flux Wave #####################################
    def prepare_data(self, W, P, flux, wave, para, fix_CO=False):
        self.W = self.Ws[W]
        self.P = self.Ps[P]
        self.nwave = wave        
        index = self.get_flux_in_Prange(para, fix_CO=fix_CO)
        flux  = flux[index]


        # flux, wave = self.get_flux_in_Wrange(flux, wave)
        print(f"flux: {flux.shape[0]}, wave: {len(self.nwave)}")
        self.name = f"{self.Names[P]} in {W}"

        #gpu only
        flux = cp.asarray(flux, dtype=cp.float32)
        wave = cp.asarray(wave, dtype=cp.float32)
        self.flux = cp.clip(-flux, 0.0, None)
        self.wave = wave
        self.nw = len(wave)
    
    def get_flux_in_Prange(self, para, fix_CO=True):
        Fs, Ts, Ls = self.P
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

    def plot_v(self, vs, idx, nidx=None, c=None, ax=None):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,5))[1]
        vs = cp.asnumpy(vs)
        v = vs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)
    
    def plot_nv(self, nvs, idx, nidx=None, c="k", ax=None, fineW=1):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,1))[1]
        v = nvs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)
        if fineW: self.get_wave_axis(ax=ax)

    def init_pcp(self, step=0.3):
        self.v = self.get_eigv(self.flux, top=200, out_w=False)
        self.nv = cp.asnumpy(self.v)
        self.plot_V(self.nv, step=step)
        plt.ylabel(self.name)
####################################### Mask #######################################
    def plot_mask(self, mask, ymin=0, ymax=0.3, c='r', lw=0.2, ax=None):
        ax = ax or plt.gca()
        ax.vlines(self.nwave[mask], ymin=ymin, ymax=ymax, color=c, lw=lw)

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
        peaks, prop = find_peaks(nvv, prominence=(prom, None))
        return peaks, prop, nvv

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
            ax.set_xticks(np.arange(int(wave[0]), np.ceil(wave[-1]), 200))
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

    def pcp_transform(self, MLv, MSv, NLv, NSv, top=15):
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

    def get_Xname(self, top=15):
        name = []
        for X in ["M", "N"]:
            for XX in ["L", "S"]:
                name += [f"{X}{XX}{i}" for i in range(top)]
        return name

    def pcp_np(self, save=0):
        self.nXv = cp.asnumpy(self.Xv)
        self.npcpFlux = cp.asnumpy(self.pcpFlux)
        if save:
            PATH= "/scratch/ceph/swei20/data/dnn/BHB/bosz_pcp.h5"
            self.save_dnn_pcp(PATH)

    def save_dnn_pcp(self, DNN_PCP_PATH):
        ww = self.W[3][:1]
        with h5py.File(DNN_PCP_PATH, 'a') as f:
            f.create_dataset(f"flux{ww}", data=self.npcpFlux, shape=self.npcpFlux.shape)
            f.create_dataset(f"pcp{ww}", data=self.nXv, shape=self.nXv.shape)
            f.create_dataset(f"para{ww}", data=self.dfpara.values, shape=self.dfpara.shape)
            f.create_dataset(f"pc{ww}", data=self.nv, shape=self.nv.shape)



        

    def pcp_ntransform(self, MLv, MSv, NLv, NSv, out=0):
        self.npcp20 = np.vstack([MLv[:5],MSv,NLv[:5],NSv])
        flux = cp.asnumpy(self.flux)
        nflux20 = np.dot(flux, self.pcp20.T)
        for i in range(20):
            self.dfpara[f"p{i}"] = nflux20[:,i]
        if out:
            return nflux20, npcp20

    # def pcp_transform(self, MLv, MSv, NLv, NSv, out=0):
    #     self.pcp20 = cp.vstack([MLv[:5],MSv,NLv[:5],NSv])
    #     flux20 = cp.dot(self.flux, self.pcp20.T)
    #     nflux20 = cp.asnumpy(flux20)
    #     npcp20 = cp.asnumpy(pcp20)
    #     for i in range(20):
    #         self.dfpara[f"p{i}"] = nflux20[:,i]
    #     if out:
    #         return nflux20, npcp20

# PCP20_PATH = '/scratch/ceph/szalay/swei20/AE/PCP_FLUX_LL20.h5'
    def save_PCP(self, PCP20_PATH, nflux20, npcp20):
        with h5py.File(PCP20_PATH, 'w') as f:
            f.create_dataset("flux", data=nflux20, shape=nflux20.shape)
            f.create_dataset("pcp", data=npcp20.values, shape=npcp20.shape)
            f.create_dataset("wave", data=nwave, shape=nwave.shape)
            f.create_dataset("para", data=self.dfpara.values, shape=self.dfpara.shape)



#         with h5py.File(PCP20_PATH, 'r') as f:
#             flux20 = f["flux20"][()]
#             para = f["para"][()]
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



    def set_unique_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
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

