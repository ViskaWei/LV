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
        self.center = False
        self.prod = None 
        self.nf = None
        self.nw = None
        ################################ RPCA ###############################
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

    
################################ Flux Wave #####################################
    def prepare_data(self, flux, wave, T, W, fix_CO=True, para=None, cleaned=False, center=True, save=False):
        if cleaned: 
            self.flux, self.mean, self.wave = flux, None, wave
        else:
            self.center = center
            self.flux, self.mean, self.wave = self.init_flux_wave(flux, wave, para, T, W, fix_CO=fix_CO)
        self.nf, self.nw = self.flux.shape
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
            return flux, None, wave

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
    def prepare_model(self, la=10.0, svd_tr=50, ep=100, rate=0.15, prll=0):
        self.svd_tr = svd_tr
        self.epoch = ep 
        self.init_model_params(la, rate)
        # self.tol = self.init_tolerance()
        print(f"lambda {self.la} | rate {self.rate} | gs {self.gs:.3f} | gl {self.gl:.2f} | ep {self.epoch} | svd {self.svd_tr}")

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
        mask = self.init_mask(rank=10, ratio=0.2)
        S = self.flux[:,  mask]
        # L = self.flux[:, ~mask]
        L = np.zeros((self.nf, self.nw))
        R = np.zeros((self.nf, self.nw))
        return S, L, R, mask

    def init_mask(self, rank=10, ratio=0.2):
        _, w, v = self._svd(self.flux, rank=rank)
        vv = np.abs(v[0])
        vmax = np.max(vv)
        mask = (vv > ratio * vmax)
        return mask

    def update_L(self, L):
        return self.shrink_mat(L, self.gl)

    def update_mask(self, mask):

        self.mask = mask
        pass

    

    def shrink_vec(self, vec, cutoff):
        return np.sign(vec) * np.maximum((np.abs(vec) - cutoff), 0.0)

    def shrink_mat(self, mat, cutoff):
        u, w, v = self._svd(mat, self.svd_tr, tol=1e-2)
        prox_w = np.maximum(w - cutoff, 0.0) 
        self.L_eigs = prox_w[prox_w > 0.0]
        # print(self.L_eigs[:10])
        self.L_rk = len(self.L_eigs)
        u, prox_w, v = u[:,:self.L_rk], prox_w[:self.L_rk], v[:self.L_rk, :]
        print(f"L_{self.L_rk}", end=" ") 
        return (u * prox_w).dot(v)

    def _svd(self, X, rank=40):
        u, s, v = svds(X, k=rank, tol=1e-2)
        return  u[:,::-1], s[::-1], v[::-1, :]
    


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
            if self.stop(ep):
                break
        t= time.time() - start
        print(f"t: {t:.2f}")
        self.h["ep"] = ep + 1
        self.finish(*args)

    def episode(self, ep, *args):
        S, L, R, mask = args
        R, S, L, U = self.update_RSLU(R, S, L, U)
        RSL = np.hstack((R, S, L))
        E = (self.flux - R - S - L) / 3.0
        new_z = RSL + np.tile(E, (1, 3))

        print(f"{loss:.2e}", end="")

        if self.L_rk < self.L_lb:
            self.h['loss'][ep]  = self.loss(R, S, L)

        return R, S, L, U, new_z


