import sys
import time
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from numpy.linalg import svd, norm
from scipy.sparse.linalg import svds
from multiprocessing.pool import ThreadPool
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
        #     gs_max = 1.14 # precompute np.abs(flux).max() 9.816 for all
        #     gl_max = 91.0 # precompute norm(flux, ord=2)
        # else:
        #     gs_max = 7.8
        #     gl_max = 3175.0
        self.gs = None
        self.gl = None
        self.rate = None

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
        l2 = np.sum(self.flux**2)**0.5
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
        self.tol = self.init_tolerance()
        print(f"lambda {self.la} | rate {self.rate} | gs {self.gs:.3f} | gl {self.gl:.2f} | ep {self.epoch} | svd {self.svd_tr}")

        if prll: 
            self.pool = ThreadPool(processes=self.N) # Create thread pool for asynchronous processing

    def init_model_params(self, la, rate):
        self.la = la
        self.rate = rate
        if self.center:
            self.gs = 0.05* self.GCs[self.g][0]
            self.gl = self.rate * self.GCs[self.g][-1]
        else:
            self.gs = 0.005 * self.rate * self.Gs[self.g][0]
            self.gl = self.rate * self.Gs[self.g][-1]
        # self.gl = 6
        self.rho = 1.0 / self.la

    def init_tolerance(self):
        abs_tol   = 1e-4 * np.sqrt(self.nf * self.nw * self.N)
        rel_tol   = 1e-2
        rel_tol_N = 1e-2 * np.sqrt(self.N)
        svd_tol   = 1e-3
        return [abs_tol, rel_tol, rel_tol_N, svd_tol]

################################ RPCA #####################################

    def init_pcp(self):
        R = np.zeros((self.nf, self.nw))    #Residual
        S = np.zeros((self.nf, self.nw))    #Sparse components
        L = np.zeros((self.nf, self.nw))    #Low rank components
        U = np.zeros((self.nf, self.nw))
        z = np.zeros((self.nf, self.nw * self.N))

        h = {}
        h['loss'] = np.zeros(self.epoch)
        h['res'] = np.zeros(self.epoch)
        h['dz'] = np.zeros(self.epoch)
        h['eps_r']  = np.zeros(self.epoch)
        h['eps_U'] = np.zeros(self.epoch)
        self.h = h
        return R, S, L, U, z

    def loss(self, R, S, L):
        noise   = norm(R, ord='fro') ** 2    # squared frobenius norm (makes X_i small)
        sparse  = self.gs * self.S_l1 # L1 norm (makes X_i sparse)
        lowrank = self.gl * self.L_eigs.sum() # nuclear norm
        # lowrank = self.gl * norm(L, ord="nuc")         # nuclear norm (makes X_i low rank)
        # print(f"loss R {noise:.2f} S {sparse:.2f} L {lowrank:.2f}")
        return noise + sparse + lowrank

################################ RUN #####################################


    def episode(self, ep, *args):
        R, S, L, U, z = args
        # print(R.shape, S.shape, L.shape, U.shape)
        R, S, L, U = self.update_RSLU(R, S, L, U)
        RSL = np.hstack((R, S, L))
        E = (self.flux - R - S - L) / 3.0
        new_z = RSL + np.tile(E, (1, 3))

        if self.L_rk < self.L_lb:
            self.h['loss'][ep]  = self.loss(R, S, L)
            self.h['res'][ep]   = self.norm_fn(E, ord="fro", s=3)
            self.h['dz'][ep]    = self.norm_fn(self.rho * (new_z - z), ord="fro", s=1)
            self.h['eps_r'][ep] = self.tol[0] + self.tol[1] * np.maximum(self.norm_fn(RSL, 'fro'), self.norm_fn(new_z, 'fro'))
            self.h['eps_U'][ep] = self.tol[0] + self.tol[2] * self.norm_fn(self.rho * U, 'fro')
        # self.h['res'][ep]   = (3.0 * norm(E, 'fro')**2)**0.5
        # self.h['dz'][ep]    = norm(self.rho * (new_z - z), 'fro')
        # self.h['eps_r'][ep] = self.tol[0] + self.tol[1] * np.maximum(norm(RSL, 'fro'), norm(new_z, 'fro'))
        # self.h['eps_U'][ep] = self.tol[0] + self.tol[2] * norm(self.rho * U, 'fro')

        return R, S, L, U, new_z

    def norm_fn(self, X, ord="fro", s=1):
        return (s * np.sum(X**2.0))**0.5

    def stop(self, ep):
        if (self.L_rk <=2) : return True
        small_res = (self.h['res'][ep] < self.h['eps_r'][ep])
        small_dz  = (self.h['dz'][ep] < self.h['eps_U'][ep])
        return small_res and small_dz
        
    def pcp(self):
        start = time.time()
        args = self.init_pcp()
        # print("\n=====Starting RPCA=====")
        # print("\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s" %('iter', 'res', 'eps r', 
        #                                     'dz', 'eps U', 'loss'))
        for ep in range(self.epoch):
            print(f"|EP{ep+1}_", end="")
            args = self.episode(ep, *args)

            if (self.L_rk < self.L_lb):
            # if (ep == 0) or (np.mod(ep + 1,10) == 0):
                print("\n")
                print("%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f" %(ep + 1,
                        self.h['res'][ep], self.h['eps_r'][ep], self.h['dz'][ep], 
                        self.h['eps_U'][ep], self.h['loss'][ep]))
            if self.stop(ep):
                break
        self.h["t"] = time.time() - start
        print(f"t: {self.h['t']:.2f}")
        self.h["ep"] = ep + 1
        self.finish(*args)

################################ Updates #####################################
    def update_R(self, R, B=None):
        return (R - B) / (1.0 + self.la)

    def update_S(self, S, B=None):
        S = self.prox_l1((S - B), (self.la * self.gs))
        self.get_sparse_ratio(S)
        return S


    def get_sparse_ratio(self, S , out=0):
        SSum = np.sum(np.abs(S),  axis=0)
        self.S_l1 = np.sum(SSum)
        # if (self.L_rk < self.L_lb) and (self.S_l1 > self.S_lb):
        SMaxs =SSum[SSum  >  self.clp * np.max(SSum)]
        print(f"S_{len(SMaxs)}", end=" ")
        if out: return SSum, SMaxs

    def update_L(self, L, B=None):
        return self.prox_mat((L - B), (self.la * self.gl))

    def prox_l1(self, mat, cutoff):
        return np.maximum(0, mat - cutoff) - np.maximum(0, -mat - cutoff)

    def prox_mat(self, mat, cutoff):
        u, s, vt = svds(mat, k=self.svd_tr, tol=self.tol[3])
        u, s, v = u[:,::-1], s[::-1], vt[::-1, :]
        # u, s, v = svd(mat, full_matrices=False)
        prox_s = self.prox_l1(s, cutoff)
        self.L_eigs = prox_s[prox_s > 0.0]
        # print(self.L_eigs[:10])
        self.L_rk = len(self.L_eigs)
        u, prox_s, v = u[:,:self.L_rk], prox_s[:self.L_rk], v[:self.L_rk, :]
        print(f"L_{self.L_rk}", end=" ") 
        return (u * prox_s).dot(v)

    def update_RSLU(self, R, S, L, U):
        B = (R + S + L - self.flux) / 3.0 + U
        if self.pool is not None:
            R, S, L = self.update_RSL_parallel(R, S, L, B, self.pool)
        else:
            R = self.update_R(R, B)
            L = self.update_L(L, B)
            S = self.update_S(S, B)
        return R, S, L, B

    def update_RSL_parallel(self, R, S, L, B, pool): 
        async_R = pool.apply_async(lambda x: self.update_R(x, B), [R])
        async_S = pool.apply_async(lambda x: self.update_S(x, B), [S])
        async_L = pool.apply_async(lambda x: self.update_L(x, B), [L])

        R = async_R.get()
        S = async_S.get()
        L = async_L.get()
        return R, S, L


################################ Finishes #####################################

    def finish(self, *args):
        self.R = args[0]
        self.S = args[1]
        self.L = args[2]
        
        self.SSum = np.mean(np.abs(self.S), axis=0)
        self.SMax = np.max(self.SSum)

        self.SClp = np.maximum(self.SSum - self.clp * self.SMax, 0.0)
        self.sdx  = self.wave[self.SClp > 0.0]
        if self.save: self.save_results()

    
    def plot_S(self, wave=None, S = None, SClp=None, ax=None, ):
        wave = self.wave if wave is None else wave
        S = self.S if S is None else S
        SClp = self.SClp if SClp is None else SClp

        if ax is None: ax = plt.gca()
        if SClp is not None: 
            ax.plot(wave, SClp, c="r")
            ax.plot(wave, -np.mean(np.abs(S), axis=0), c="b")
        else:
            ax.plot(wave, np.mean(np.abs(S), axis=0), c="r")


    def eval_pcp(self, roff=3000, soff=2000):
        res = self.R + self.S + self.L - self.flux
        plt.plot(self.wave, -res.sum(0)-roff, label=f"Res - {roff}", c='g')
        plt.plot(self.wave, abs(self.L).sum(0), label = "LowRank", c='skyblue')
        plt.plot(self.wave, self.S.sum(0)-soff, label=f"Sparse - {soff}", c='r')
        plt.xlabel("wave")
        plt.ylabel("norm flux")
        plt.title(f"Spec {self.nf}  @ LowT 3500 -6500K")
        plt.legend()

    def save_results(self, PATH=None):
        PATH = PATH or f"../data/{self.g}.h5"
        print("=================SAVINHG==================")
        with h5py.File(PATH, 'w') as f:
            # f.create_dataset('R', data = self.R)
            f.create_dataset('wave', data = self.wave)
            f.create_dataset('S', data = self.S)
            f.create_dataset('L', data = self.L)
            f.create_dataset('sdx', data = self.sdx)

 
    @staticmethod
    def avg(*args):
        return np.mean([*args], axis=0)


    def plot_pcp(self, u, s, v, wave, cmap="hot"):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(self.flux), cmap=cmap,)
        plt.title("Original Flux")
        plt.subplot(2, 2, 2)
        plt.imshow(np.abs(self.L), cmap=cmap)
        plt.title("Low rank matrix")
        plt.subplot(2, 2, 3)
        plt.imshow(np.abs(self.S), cmap=cmap, )
        plt.title("Sparse matrix")
        plt.subplot(2, 2, 4)
        for i in range(min(len(v),5)):
            plt.plot(wave, v[i] + 0.3*(1 + i))
        plt.plot(wave, np.mean(np.abs(S), axis=0), c="k")
        # plt.imshow(np.dot(u, np.dot(np.diag(s), v)), cmap="gray")
        plt.title("L & S")
        plt.show()