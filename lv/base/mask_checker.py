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
from lv.pcp.pcpc import pcp_cupy
import h5py
from tqdm import tqdm
import seaborn as sns




class Mask_checker(object):
    def __init__(self):
        ################################ Flux Wave ###############################
        self.Ws = {"Blue": [3800, 6500, 2300], "RedL": [6300, 9700, 3000], "RedM": [7100, 8850, 5000],
                   "NIR": [9400, 12600, 4300]}

        self.Ps = {"M": [[-2.5, 0.0], [3500, 5000], [0.0, 2.0]], 
                   "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0]],
                   "C": [[-2.0, 0.0], [4500, 6000], [4.0, 5.0]],
                   "B":  [[-2.5,-1.5], [7000, 9500], [2.0, 3.0]],
                   "R":  [[-1.0, 0.0], [5000, 6500], [2.0, 3.0]],
                   "G":  [[-2.5,-1.0], [3500, 5500], [0.0, 3.5]]}

        self.Mps= {"M": [ 100, 0.6, 0.2], 
                    "W": [ 100, 0.6, 0.2],
                    "C": [ 100, 0.6, 0.2],
                    "B": [ 100, 0.6, 0.2],
                    "R": [ 100, 0.6, 0.2],
                    "G": [ 100, 0.6, 0.2]}

        self.Msks = {}
        self.Nms = {"M": "M31 Giant", "W": "MW Warm", "C": "MW Cool", "B": "BHB", "R": "RHB", "G":"DwarfG Giant"}
        self.Flux = {}
        self.nFlux = {}
        self.Size = {}
        self.Vs = {}
        self.nVs = {}

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

################################ Flux Wave #####################################
    def prepare_data(self, W, flux, wave, para, fix_CO=False):
        self.W = self.Ws[W]
        path = f'/scratch/ceph/szalay/swei20/AE/norm_flux_{W}_R{self.W[2]}.h5'

        # if flux is None:
        #     with h5py.File(path, 'r') as f:
        #         flux = f['flux'][()]
        #         para = f['para'][()]
        #         wave = f['wave'][()]
        #cpu only
        flux = np.clip(-flux, 0.0, None)
        self.nwave = wave        

        for p, pvals in self.Ps.items():
            index = self.get_flux_in_Prange(para, pvals, fix_CO=fix_CO)
             # flux, wave = self.get_flux_in_Wrange(flux, wave)
            flux_p = flux[index]
            self.nFlux[p] = flux_p
            self.Size[p] = flux_p.shape[0]
            print(f"# {p} flux: {self.Size[p]}, wave {W}: {wave.shape} ")

    def prepare_svd(self):
        #gpu only
        for p, flux_p in self.nFlux.items():
            self.Flux[p] = cp.asarray(flux_p, dtype=cp.float32)
        self.wave = cp.asarray(self.nwave, dtype=cp.float32)

        for p, flux_p in tqdm(self.Flux.items()):
            Vs = self.get_eigv(flux_p, top=200)
            self.Vs[p] = Vs
            self.nVs[p] = cp.asnumpy(Vs)

    
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


  
####################################### SVD #######################################

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

            
    def plot_Vs_p(self, p, top=5, step=0.3, ax=None):
        wave = self.nwave
        nv = self.nVs[p]
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        for i in range(min(len(nv),top)):
            ax.plot(wave, nv[i] + step*(i+1))
        ax.set_ylabel(f"{self.Nms[p]}")
        self.get_wave_axis(wave=wave, ax=ax)

    # def plot_eigv(self, v, top=5, mask=None, step=0.3, mask_c="r", lw=0.5, name=None, wave=None, ax=None):
    #     if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
    #     nv = cp.asnumpy(v[:top])
    #     if wave is None: wave = self.nwave
    #     for i in range(min(len(nv),top)):
    #         ax.plot(wave, nv[i] + step*(i+1))
    #     ax.set_ylabel(f"Top {top} Eigvs of {name}")
    #     self.get_wave_axis(wave=wave, ax=ax)
    #     if mask is not None: self.plot_mask(mask, ax=ax, c=mask_c, lw=lw)


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

    def plot_peaks(self):
        f, axs = plt.subplots(6,1, figsize=(16,20),facecolor="w")
        for i, (p, nv) in enumerate(self.nVs.items()):
            ax = axs[i]
            k, q, prom = self.Mps[p]
            peaks, prop, nvv = self.get_peaks_p(p, k, q, prom) 
            self.Msks[p] = self.plot_mask_from_peaks(p, peaks, prop, nvv, ax=ax)
            ax.set_title(f"k {k} q {q} prom {prom}")


    def get_mask_from_peaks(self, peaks, prop):
        prom = prop["prominences"]
        lb = prop["left_bases"]
        ub = prop["right_bases"]
        mask = np.zeros_like(self.nwave, dtype=bool)

        for i in range(len(peaks)):
            mask[lb[i]:ub[i]+1] = True
        return mask, np.max(prom)

    def plot_mask_from_peaks(self, p, peaks, prop, nvv, ax=None):
        mask, pmax = self.get_mask_from_peaks(peaks, prop)
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.plot(self.nwave, nvv, c="k")
        ax.plot(self.nwave[peaks], nvv[peaks], "bx", markersize=10)
        ax.vlines(self.nwave[mask], ymin=-pmax*0.1, ymax=0.0, color="r", alpha=0.5)
        self.get_wave_axis(ax=ax)
        ax.set_ylabel(f"{self.Nms[p]}")
        return mask


    def plot_mask(self, ax=None, c="r", ymin=0.5, ymax=1, alpha=0.8):
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        for i in range(len(self.lb)):
            ax.axvspan(self.nwave[self.lb[i]], self.nwave[self.ub[i]], ymin=ymin, ymax=ymax, color=c, alpha=alpha)



    # def plot_peaks(self, nvv, peaks, k, prom, ax=None):
    #     if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
    #     ax.plot(self.nwave, nvv, c="k", label=f"leverage score k={k}")
    #     vpeaks = nvv[peaks]
    #     ax.plot(self.nwave[peaks], vpeaks, "bx", markersize=10, label=f"prominence={prom}")
    #     if self.nmask is not None: self.plot_masked(ax=ax)
    #     self.get_wave_axis(ax=ax)
    #     ax.legend()
        


    def plot_complement(self, ax=None, c="g", ymin=0.0, ymax=0.1, alpha=0.8):
        if ax is None: ax = plt.subplots(1, figsize=(16,3),facecolor="w")[1]
        ax.axvspan(self.nwave[0], self.nwave[self.lb[0]]-1, ymin=ymin, ymax=ymax, color=c, alpha=alpha)

        for i in range(1, len(self.lb)):
            ax.axvspan(self.nwave[self.ub[i-1]]+1, self.nwave[self.lb[i]]-1, ymin=ymin, ymax=ymax, color=c, alpha=alpha)

        ax.axvspan(self.nwave[self.ub[-1]]+1, self.nwave[-1], ymin=ymin, ymax=ymax, color=c, alpha=alpha)

    def plot_MN_mask(self, v=None):
        f, ax = plt.subplots(1, figsize=(16,3),facecolor="w")
        self.plot_complement(ymin=0,ymax=0.5, ax=ax, alpha=0.5)
        self.plot_masked(ymin=0.5,ymax=1, ax=ax, alpha=0.5)
        if v is not None: self.plot_v(v, 0, ax=ax, c= 'k')
        self.get_wave_axis(ax=ax, xgrid=0)

####################################### M N #######################################

    def get_MN(self, mask, top=5):
        M = cp.zeros_like(self.flux)
        N = cp.zeros_like(self.flux)
        M[:,  mask] = self.flux[:,  mask]
        N[:, ~mask] = self.flux[:, ~mask]    
        self.M = M
        self.N = N    
        self.Mv =self.get_eigv(M, top=top)
        self.Nv =self.get_eigv(N, top=top)

    def plot_MN(self, step=0.3, axs=None):
        if axs is None:
            axs = plt.subplots(2,1,figsize=(16,10))[1]
        self.plot_eigv(self.Mv, mask=None, name="M", step=step, ax=axs[0])
        self.plot_masked(ax=axs[0], ymin=0, ymax=0.1)
        axs[0].set_ylim(0.0, None)
        self.plot_eigv(self.Nv, mask =None, name="N", mask_c="k", step=step, ax=axs[1])
        self.plot_complement(ax=axs[1], ymin=0, ymax=0.1)
        axs[1].set_ylim(0.0, None)


####################################### PCP #######################################
    def _pcp(self, X, delta=1e-6, mu=None, lam=None, norm=None, maxiter=50):
        XL, XS, (_,_,XLv) = pcp_cupy(X, delta=delta, mu=mu, lam=lam, norm=norm, maxiter=maxiter)
        XSv = self.get_eigv(XS, top = 5)
        return XL, XS, XLv, XSv

    def pcp_transform(self, MLv, MSv, NLv, NSv, out=0):
        pcp20 = cp.vstack([MLv[:5],MSv,NLv[:5],NSv])
        flux20 = cp.dot(self.flux, pcp20.T)
        nflux20 = cp.asnumpy(flux20)
        for i in range(20):
            self.dfpara[f"p{i}"] = nflux20[:,i]
        if out:
            return nflux20

# PCP20_PATH = '/scratch/ceph/szalay/swei20/AE/PCP_FLUX_LL20.h5'
    def save_pcp_flux(self, PCP20_PATH, nflux20):
        with h5py.File(PCP20_PATH, 'w') as f:
            f.create_dataset("flux20", data=nflux20, shape=nflux20.shape)

#         with h5py.File(PCP20_PATH, 'r') as f:
#             flux20 = f["flux20"][()]
#             para = f["para"][()]
# ####################################### M LS #######################################
    


    # def get_flux_stats(self, flux=None):
    #     flux = flux or self.flux
    #     # return cp.max(cp.sum(cp.abs(self.flux), axis=1)), cp.sum(self.flux**2)**0.5
    #     l1_inf = cp.max(cp.abs(self.flux))
    #     l2 = cp.sum(self.flux ** 2) ** 0.5
    #     print(f"l1_inf: {l1_inf}, l2: {l2}")
    #     if self.center:
    #         self.GCs[self.g] = [l1_inf, l2]
    #     else:
    #         self.Gs[self.g] = [l1_inf, l2]


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



