import time
# import fbpca
import logging
import numpy as np
import cupy as cp
import pandas as pd
# from scipy.sparse.linalg import svds
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from tqdm import tqdm
import seaborn as sns




class PCA(object):
    def __init__(self):
        ################################ Flux Wave ###############################
        self.Ws = {"Blue": [3800, 6500, 2300, "Blue"], "RedL": [6300, 9700, 3000, "RedL"], "RedM": [7100, 8850, 5000, "RedM"],
                   "NIR": [9400, 12600, 4300, "NIR"]}

        self.Ps = {"M": [[-2.5, 0.0], [3500, 5000], [0.0, 2.0]], 
                   "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0]],
                   "C": [[-2.0, 0.0], [4500, 6000], [4.0, 5.0]],
                   "B":  [[-2.5,-1.5], [7000, 9500], [2.0, 3.0]],
                   "R":  [[-1.0, 0.0], [5000, 6500], [2.0, 3.0]],
                   "G":  [[-2.5,-1.0], [3500, 5500], [0.0, 3.5]]}

        self.Mps_Blue ={"M": [ 100, 0.6, 0.2], 
                        "W": [ 100, 0.6, 0.2],
                        "C": [ 100, 0.6, 0.2],
                        "B": [ 100, 0.6, 0.2],
                        "R": [ 100, 0.6, 0.2],
                        "G": [ 100, 0.6, 0.2]}

        self.Mps_RedM ={"M": [ 5, 0.7, 0.02], 
                        "W": [5, 0.7, 0.02],
                        "C": [5, 0.7, 0.02],
                        "B": [ 5, 0.7, 0.02],
                        "R": [ 5, 0.7, 0.02],
                        "G": [ 5, 0.7, 0.02]}
        self.Mps = {"RedM": self.Mps_RedM, "Blue": self.Mps_Blue}
        self.Msks = {}
        self.nMsks = {}

        self.Nms = {"M": "M31 Giant", "W": "MW Warm", "C": "MW Cool", "B": "BHB", "R": "RHB", "G":"DwarfG Giant"}
        self.Flux = {}
        self.nFlux = {}
        self.pcFlux = {}
        self.npcFlux = {}
        self.Size = {}
        self.Vs = {}
        self.nVs = {}
        self.Ms = {}
        self.Ns = {}
        self.Mvs = {}
        self.Nvs = {}
        self.wave = None
        self.nwave = None
        self.mean = None
        self.size = None
        self.center = False
        self.prod = None 
        self.nf = None
        self.nw = None
        self.mask = None
        self.nmask = None

        self.cmap="YlGnBu"
        self.color = {"T": "gist_rainbow", "L": "turbo", "F": "plasma", "C": "gist_rainbow", "O":"winter"}
        self.ps =  [["p0","p1", "p2", "p3", "p4"],["p5","p6", "p7", "p8", "p9"],["p10","p11", "p12", "p13", "p14"],["p15","p16", "p17", "p18", "p19"]]
        self.name = None
        self.lick = None

####################################### Flux #####################################
    def prepare_data(self, flux, wave, para, W=None, fix_CO=False):
        # flux = np.clip(-flux, 0.0, None)
        if W is not None: self.W = self.Ws[W]
        self.nwave = wave        

        for p, pvals in self.Ps.items():
            index = self.get_flux_in_Prange(para, pvals, fix_CO=fix_CO)
             # flux, wave = self.get_flux_in_Wrange(flux, wave)
            flux_p = flux[index]
            self.nFlux[p] = flux_p
            self.Size[p] = flux_p.shape[0]
            print(f"# {p} flux: {self.Size[p]}, wave {W}: {wave.shape} ")

    def prepare_svd(self, top=200):
        #gpu only
        for p, flux_p in self.nFlux.items():
            self.Flux[p] = cp.asarray(flux_p, dtype=cp.float32)
        self.wave = cp.asarray(self.nwave, dtype=cp.float32)

        for p, flux_p in tqdm(self.Flux.items()):
            Vs = self.get_eigv(flux_p, top=200)
            self.Vs[p] = Vs
            self.pcFlux[p] = self.Flux[p].dot(Vs.T)
            self.npcFlux[p] = cp.asnumpy(self.pcFlux[p])
            self.nVs[p] = cp.asnumpy(Vs)

    def save_PCA(self, PATH):
        with h5py.File(PATH, "w") as f:
            for p, nV in self.nVs.items():
                f.create_dataset(f"pc{p}", data=nV, shape=nV.shape)
                f.create_dataset(f"pcFlux{p}", data=self.npcFlux[p], shape=self.npcFlux[p].shape)

    def get_flux_in_Prange(self, para, p, fix_CO=True):
        Fs, Ts, Ls = p
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
        return self.dfpara.index

    def get_flux_in_Wrange(self, flux, wave):
        Ws = self.W
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]

    def init_para(self, para):
        return pd.DataFrame(data=para, columns=["F","T","L","C","O"])

    def get_wave_axis(self, wave= None, ax=None, xgrid=True):
        if wave is None: 
            ax.set_xlim(self.nwave[0]-1, self.nwave[-1]+2)
            ax.set_xticks(np.arange(self.W[0], self.W[1], 200))
            
        else:
            ax.set_xlim(wave[0]-1, wave[-1]+2)
            ax.set_xticks(np.arange(int(wave[0]), np.ceil(wave[-1]), 200))
        ax.xaxis.grid(xgrid)
####################################### SVD  #######################################

    def _svd(self, X):
        return cp.linalg.svd(X, full_matrices=0)

    def get_eigv(self, X, top=5):
        _,_,v = self._svd(X)
        return v[:top] 

    def plot_Vs(self, top=5, step=0.3):
        wave = self.nwave
        f, axs = plt.subplots(6,1, figsize=(16,18),facecolor="w")
        for i, (p, nv) in enumerate(self.nVs.items()):
            ax = axs[i]
            self.plot_Vs_p(p, ax=ax, top=top, step=step)
            
    def plot_Vs_p(self, p, vs=None, top=5, step=0.3, ax=None):
        wave = self.nwave
        if vs is None: vs = self.nVs
        nv = self.nVs[p]
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(wave, nv[i] + step*(i+1))
        ax.set_ylabel(f"{self.Nms[p]}")
        self.get_wave_axis(wave=wave, ax=ax)

    def plot_V(self, nv, top=5, step=0.3, ax=None):
        wave = self.nwave        
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(wave, nv[i] + step*(i+1))
        self.get_wave_axis(wave=wave, ax=ax)
####################################### Mask #######################################
    def get_mask_from_nv(self, v, k=5, q=0.8):
        vv = np.sum(v[:k]**2, axis=0)
        cut = np.quantile(vv, q)
        mask = vv > cut        
        return mask, vv

    def get_peaks_p(self, p, k=100, q=0.6, prom=0.2):
        nv = self.nVs[p]
        mask, nvv = self.get_mask_from_nv(nv, k=k, q=q)   
        nvv[~mask]  = 0.0
        peaks, prop = find_peaks(nvv, prominence=(prom, None))
        return peaks, prop, nvv

    def plot_peaks_p(self, p, k, q, prom, ax=None):
        peaks, prop, nvv = self.get_peaks_p(p, k, q, prom)
        vpeaks = nvv[peaks]
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.plot(self.nwave, nvv, c="k", label=f"lvrg {k}")
        ax.plot(self.nwave[peaks], vpeaks, "bx", markersize=10, label=f"prom {prom}")
        ax.set_ylabel(f"{self.Nms[p]}")
        self.get_wave_axis(ax=ax)
        ax.legend()
        return peaks, prop, nvv

    def get_masks(self):
        f, axs = plt.subplots(6,1, figsize=(16,10),facecolor="w")
        for i, (p, nv) in enumerate(self.nVs.items()):
            ax = axs[i]
            Mps = self.Mps[self.W[3]]
            k, q, prom = Mps[p]
            peaks, prop, nvv = self.get_peaks_p(p, k, q, prom) 
            self.nMsks[p] = self.plot_mask_from_peaks(p, peaks, prop, nvv, ax=ax)
            ax.annotate(f"{self.Nms[p]}\nk={k} q={q} prom={prom}", (0.5, 0.5), xycoords="axes fraction")
         
    def get_mask_from_peaks(self, peaks, prop):
        prom = prop["prominences"]
        lb = prop["left_bases"]
        ub = prop["right_bases"]
        mask = np.zeros_like(self.nwave, dtype=bool)

        for i in range(len(peaks)):
            mask[lb[i]:ub[i]+1] = True
        self.pMaxs = np.max(prom)
        return mask

    def plot_mask_from_peaks(self, p, peaks, prop, nvv, ax=None):
        mask = self.get_mask_from_peaks(peaks, prop)
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.plot(self.nwave, nvv, c="k")
        ax.plot(self.nwave[peaks], nvv[peaks], "bx", markersize=10)
        ax.vlines(self.nwave[mask], ymin=-self.pMaxs*0.1, ymax=0.0, color="r", alpha=0.5)
        self.get_wave_axis(ax=ax)
        return mask

    def plot_MN_mask(self, mask, idx=0, nv=None, ax=None):
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.vlines(self.nwave[mask], ymin=0.0, ymax=self.pMaxs * 0.2, color="r", alpha=0.3)
        ax.vlines(self.nwave[~mask], ymin=-self.pMaxs * 0.2, ymax=0.0, color="g", alpha=0.3)
        if nv is not None: self.plot_nv(nv, idx, ax=ax, c= 'k')
        self.get_wave_axis(ax=ax, xgrid=0)

    def plot_MN_masks(self, idx=0):
        f, axs = plt.subplots(6,1, figsize=(16,10),facecolor="w")
        for i, (p, mask) in enumerate(self.nMsks.items()):
            ax = axs[i]
            nv = self.nVs[p]
            self.plot_MN_mask(mask, idx=idx, nv=nv, ax=ax)
            ax.annotate(f"{self.Nms[p]}", (0.5, 0.5), xycoords="axes fraction")    
####################################### M N  #######################################
    def get_MN_p(self, p, top=5):
        mask = cp.asarray(self.nMsks[p], dtype=bool)
        flux = self.Flux[p]
        M = cp.zeros_like(flux)
        N = cp.zeros_like(flux)
        M[:,  mask] = flux[:,  mask]
        N[:, ~mask] = flux[:, ~mask]    
        self.Ms[p] = M
        self.Ns[p] = N    
        self.Mvs[p] =self.get_eigv(M, top=top)
        self.Nvs[p] =self.get_eigv(N, top=top)
        self.Msks[p] = mask

    def get_MNs(self):
        for i, (p, mask) in enumerate(self.nMsks.items()):
            self.get_MN_p(p, top=5)

    def plot_MNs(self,step=0.3):
        f, axs = plt.subplots(6,2, figsize=(30,18),facecolor="w")
        for i, (p, mask) in enumerate(self.nMsks.items()):
            ax = axs[i]
            self.get_MN_p(p, top=5)
            Mv = cp.asnumpy(self.Mvs[p])
            self.plot_V(Mv, top=5, step=step, ax=ax[0])
            ax[0].vlines(self.nwave[mask], ymin=0.0, ymax=self.pMaxs * 0.2, color="r", alpha=0.3)
            ax[0].set_ylabel(f"{self.Nms[p]}")
    
            Nv = cp.asnumpy(self.Nvs[p])
            self.plot_V(Nv, top=5, step=step, ax=ax[1])
            ax[1].vlines(self.nwave[~mask], ymin=-self.pMaxs * 0.2, ymax=0.0, color="g", alpha=0.3)
####################################### PCP #######################################

# PCP20_PATH = '/scratch/ceph/szalay/swei20/AE/PCP_FLUX_LL20.h5'
    def save_pcp_flux(self, PCP20_PATH, nflux20):
        with h5py.File(PCP20_PATH, 'w') as f:
            f.create_dataset("flux20", data=nflux20, shape=nflux20.shape)

    def p(self, idx1, idx2, para, large=0):
        if large:
            plt.figure(figsize=(8,6), facecolor="w")
            s=5
        else:
            plt.figure(figsize=(6,4), facecolor="w")
            s=3
        sns.scatterplot(
            data=self.dfpara,x=f"p{idx1}", y=f"p{idx2}", hue=para, marker="o", s=s, edgecolor="none",palette=self.color[para])
        plt.title(self.name)

    def pp(self, idx, para):
        sns.pairplot(
            self.dfpara,
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

    def plot_lick(self, ax=None):
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
        self.get_wave_axis(ax=ax,)
        # self.set_wv_ticks(ax, lim=True)
        # ax.set_ylabel('LICK')

    def plot_v(self, vs, idx, nidx=None, c=None, ax=None):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,5))[1]
        vs = cp.asnumpy(vs)
        v = vs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)
        
    def plot_nv(self, nvs, idx, nidx=None, c=None, ax=None):
        if nidx is None: nidx = idx
        if ax is None:
            ax = plt.subplots(figsize=(16,5))[1]
        v = nvs[idx]
        ax.plot(self.nwave, v, label=nidx, c=c)

    def set_unique_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())















