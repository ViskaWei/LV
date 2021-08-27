import time
# import fbpca
import logging
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

class RPCA(object):
    def __init__(self):
        ################################ Flux Wave ###############################
        self.Ws = {"B": [3800, 6000], "R": [6000, 9200],
                   "L": [3800, 8000], "H": [8000, 12000],
                   "T": [8200, 9500], "F": [3800, 13000]}
        self.Ts = {"L": [4000, 6500], "H": [6500, 30000],
                   "T": [8000, 9000], "F": [4000, 30000]}
        self.flux = None
        self.wave = None
        self.mean = None
        self.size = None
        self.center = False
        self.prod = None 
        self.nf = None
        self.nw = None
        ################################ RPCA ###############################
        self.X = None
        self.Xmean = None
        self.N = 3
        self.Gs = {"HH0": [1.5, 260.,  1200.], 
                   "HH1": [1.4, 260.,  270.], 
                   "LL0": [9., 7200., 6300.], 
                   "LL1": [8., 6000., 1400.],
                   "LH0": [2.3, None, 860.],
                   "LH1": [2., None, 155.],
                   "HL0": [4.5, None, 3600.],
                   "HL1": [4.3, None, 790],


                    "LH": 1.14, "LL": 1.14, "TT0": [1.5,200]} 
        self.GCs = {"LL0": [7.6, 5800., 4000.],
                   "LH0": [2.3, None, 860.],

                    "HH1": [185.9, 115.0] }
        self.g = ""

        self.gs = None
        self.gl = None
        self.rate = None
        self.mask = None
        self.la = None
        self.tol = None
        self.rho = None

        self.svd_tr = None
        self.epoch = None
        self.pool = None        

        self.R = None
        
        self.L = None
        self.L_eigs = None
        self.L_rk = None
        self.L_lb = 10

        self.S = None
        self.S_lb = 10
        self.S_l1 = None
        self.SSum = None
        self.SMax = None
        self.SClp = None
        self.Snum = None
        self.clp = 0.2
        self.norm = None

        self.cmap="YlGnBu"

    
################################ Flux Wave #####################################
    def prepare_data(self, flux, wave, T, W, fix_CO=True, para=None, cleaned=False, center=True, save=False):
        if cleaned: 
            self.flux, self.mean, self.wave = flux, flux.mean(), wave
        else:
            self.center = center
            self.flux, self.mean, self.wave = self.init_flux_wave(flux, wave, para, T, W, fix_CO=fix_CO)
        self.size = self.flux.shape
        self.nf, self.nw = self.size
        # self.mu = 0.25 * self.nf * self.nw 
        self.g = T + W + str(int(fix_CO))
        print(f"center {self.center} {self.g} flux: {self.nf}, wave: {self.nw}")
        self.save = save

    def init_flux_wave(self, flux, wave, para, T, W, fix_CO):
        flux, wave = self.get_flux_in_Wrange(flux, wave, self.Ws[W])
        flux       = self.get_flux_in_Prange(flux, para, self.Ts[T], fix_CO=fix_CO)
        if self.center:
            mean = flux.mean(0)
            return flux - mean, mean, wave
        else:
            return flux, flux.mean(), wave

    def init_para(self, para):
        return pd.DataFrame(data=para, columns=["F","T","L","C","O"])

    def get_flux_in_Wrange(self, flux, wave, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]

    def get_flux_in_Prange(self, flux, para, Ts, fix_CO=True):
        dfpara = self.init_para(para)
        if fix_CO:
            # dfpara = dfpara[(dfpara["C"] == 0.0)]
            dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["O"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        self.dfpara = dfpara[(dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1])]
        return flux[self.dfpara.index]

    def get_flux_stats(self, flux=None):
        flux = flux or self.flux
        # return np.max(np.sum(np.abs(self.flux), axis=1)), np.sum(self.flux**2)**0.5
        l1_inf = np.max(np.abs(self.flux))
        l2 = np.sum(self.flux ** 2) ** 0.5
        print(f"l1_inf: {l1_inf}, l2: {l2}")
        if self.center:
            self.GCs[self.g] = [l1_inf, l2]
        else:
            self.Gs[self.g] = [l1_inf, l2]

################################ Model #####################################
    def prepare_model(self, X=None,  rs=1.0, rl=1.0, tol=0.1, svd_tr=50, ep=100):
        self.X = np.clip(-self.flux, 0.0, None) if X is None else X
        self.Xmean = self.mean if X is None else self.X.mean() 
        self.norm  = np.sum(self.X ** 2)
        self.svd_tr = svd_tr
        self.epoch = ep 
        self.tol = tol
        self.rs = rs
        self.rl = rl
        # self.init_model_params(la, rate)
        # self.tol = self.init_tolerance()
        # print(f"lambda {self.la} | rate {self.rate} | gs {self.gs:.3f} | gl {self.gl:.2f} | ep {self.epoch} | svd {self.svd_tr}")
        print(f"lambda {self.tol} | ep {self.epoch} | svd {self.svd_tr} | rs {self.rs:.2f} | rl {self.rl:.2f} |")

        # if prll: 
        #     self.pool = ThreadPool(processes=self.N) # Create thread pool for asynchronous processing

    def init_model_params(self, la, rate):
        self.la = la
        self.rate = rate
        if self.center:
            self.gs = 0.1 * self.rate * self.GCs[self.g][0]
            self.gl = self.rate * self.GCs[self.g][-1]
        else:
            self.gs = self.rate * self.Gs[self.g][0]
            self.gl = self.rate * self.Gs[self.g][-1]
        # self.gl = 6
        self.rho = 1.0 / self.la

################################ RPCA #####################################

    def init_pcp(self):
        self.isStop= False
        u, w, v = self._svd(self.X, rank=10)

        self.gl = w[-1]
        
        self.mask = self.get_mask(v[:5], ratio=0.1)
        
        S = self.get_S_from_mask(self.X, self.mask)
        # L = self.flux[:, ~mask]
        # L = np.zeros(self.size)
        # R = np.random.uniform(0.0, self.Xmean, self.shape )
        R = np.zeros((self.nf, self.nw))
        return R, S

    def get_S_from_mask(self, X, mask):
        S = np.zeros(self.size)
        S[:, mask] = X[:, mask]
        return S
    
    def update_L_from_mask(self, L):
        L[:, mask] = 0.0
        return S

    def get_mask(self, v, ratio=0.1):
        vv = np.abs(v).sum(0)
        vmax = np.max(vv)
        self.gs = ratio * vmax
        mask = (vv > self.gs)
        return mask

    def update(self, R, S):
        L = self.update_L(self.X - S - R)
        S = self.update_S(self.X - L - R)
        loss = self.loss(self.X - L - S)
        if loss < self.tol:
            self.isStop = True
        print(f"{loss:.2e}", end="")
        return R, S

    def update_L(self, L):
        L, vL = self.shrink_l2(L, self.rl * self.gl)
        mask = self.get_mask(vL[:1], ratio=0.5)
        L[:, ~mask] = 0.0
        self.mask = np.logical_or(mask, self.mask)
        return L

    def update_S(self, S):
        prox_S = self.shrink_l1(S, self.rs * self.gs)
        self.mask = prox_S > 0.0
        self.get_S_from_mask(self.X, self.mask)
        return S

    # def update_S(self, S):
    #     Svec = np.mean(abs(S), axis=0)
    #     prox_S = self.shrink_vec(Svec, self.rs * self.gs)
    #     self.mask = prox_S > 0.0
    #     S[:, ~self.mask] = 0.0
    #     S[:, self.mask] = self.X[:, self.mask]
    #     return S

    def loss(self,R):
        err = np.sqrt(np.sum(R ** 2) / self.norm)
        return err
    # def update_mask(self, S):

    #     self.mask = mask
    #     pass

    def shrink_l1(self, X, cutoff):
        return np.sign(X) * np.maximum((np.abs(X) - cutoff), 0.0)

    def shrink_l2(self, mat, cutoff):
        u, w, v = self._svd(mat, self.svd_tr, tol=1e-2)
        # print(w)
        prox_w = np.maximum(w - cutoff, 0.0) 
        self.L_eigs = prox_w[prox_w > 0.0]
        print(self.L_eigs[:10])
        self.L_rk = len(self.L_eigs)
        print(f"L_{self.L_rk}", end=" ") 
        return self.rec_svd(u, prox_w, v, rank=self.L_rk)

    def _svd(self, X, rank=40, tol=1e-2):
        u, s, v = svds(X, k=rank, tol=tol)
        return  u[:,::-1], s[::-1], v[::-1, :]
    
    def rec_svd(self, u, w, v, rank=None):
        if rank is not None:
            u, w, v = u[:,:rank], w[:rank], v[:rank, :]
        return (u * w).dot(v)

    # def init_tolerance(self):
    #     abs_tol   = 1e-4 * np.sqrt(self.nf * self.nw * self.N)
    #     rel_tol   = 1e-2
    #     rel_tol_N = 1e-2 * np.sqrt(self.N)
    #     svd_tol   = 1e-3
    #     return [abs_tol, rel_tol, rel_tol_N, svd_tol]

################################ RUN #####################################
        
    def pcp(self):
        start = time.time()
        args = self.init_pcp()
        for ep in range(self.epoch):
            print(f"|EP{ep+1}_", end="")
            args = self.episode(ep, *args)
            if self.isStop:
                break
        t= time.time() - start
        print(f"t: {t:.2f}")

        # self.h["ep"] = ep + 1
        self.finish(*args)

    def episode(self, ep, *args):
        R, S = self.update(*args)
        # RSL = np.hstack((R, S, L))
        # E = (self.flux - R - S - L) / 3.0
        # new_z = RSL + np.tile(E, (1, 3))
        # if self.L_rk < self.L_lb:
        #     self.h['loss'][ep]  = self.loss(R, S, L)
        return R, S
    
    def finish(self, *args):
        self.R = args[0]
        self.S = args[1]
        self.L = self.X - self.R - self.S
    # def stop(self, ep):
    #     if 


################################ Eval #####################################

    def eval_LSM(self, L, S, mask, vL=None, vS=None):
        f, axs = plt.subplots(2,2, figsize=(16, 8))
        self.plot_mat(L, ax=axs[0,0])
        self.plot_mat(S, ax=axs[1,0])
        self.plot_eigv(mask, M=L, v=vL, ax=axs[0,1])
        self.plot_eigv(mask, M=S, v=vS, ax=axs[1,1])
        axs[0,0].set_ylabel("L")
        axs[1,0].set_ylabel("S")        



    def plot_mat(self, mat, ax=None):
        ax = ax or plt.gca()
        ax.matshow(mat, aspect="auto", cmap=self.cmap)

    def plot_mask(self, mask, ax=None, ymax=0.3, c='r'):
        ax = ax or plt.gca()
        ax.vlines(self.wave[mask], ymin=0, ymax=ymax, color=c)

    def plot_eigv(self, mask=None, M=None, v= None,ax=None, mask_c="r"):
        ax = ax or plt.gca()
        if v is None: 
            _,w, v =self._svd(M, rank=10) 
            print(w)
        for i in range(min(len(v),5)):
            ax.plot(self.wave, v[i] + 0.3*(i+1))
        if mask is not None: self.plot_mask(mask, ax=ax, c=mask_c)