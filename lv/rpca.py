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
    def __init__(self, la=10.0, tsvd=50, ep=100, ratio=0.15):
        ################################ Flux Wave ###############################
        self.Ws = {"B": [3800, 6000], "R": [6000, 9200],
                   "L": [3800, 8000], "H": [8000, 13000],
                   "T": [8200, 9500], "F": [3800, 13000]}
        self.Ts = {"L": [4000, 6500], "H": [7000, 30000],
                   "T": [8000, 9000], "F": [4000, 30000]}
        self.flux = None
        self.wave = None
        self.mean = None
        self.nf = None
        self.nw = None
        ################################ RPCA ###############################
        self.Gs = {"HH0": 1.16, "HL": 1.14, "LH": 1.14, "LL": 1.14, "TT0": [1.5,200]} 
                # if T=="H":
        #     gs_max = 1.14 # precompute np.abs(flux).max() 9.816 for all
        #     gl_max = 91.0 # precompute norm(flux, ord=2)
        # else:
        #     gs_max = 7.8
        #     gl_max = 3175.0
        self.N = 3
        self.ratio = ratio
        self.la = la
        self.tsvd = tsvd
        self.epoch = ep
        self.L_eigs = None
        self.L_rk = None
        self.S_l1 = None
        self.tol = None
        self.gs = None
        self.gl = None
        self.pool = None
        self.rho = None
        self.R = None
        self.S = None
        self.L = None
        self.SSum = None
        self.SMax = None
        self.SClp = None
        self.clp = 0.2
        self.Snum = None



        self.init_pcp()
################################ Flux Wave #####################################
    def prepare_data(self, flux, wave, para, T, W, fix_C0=True):
        self.flux, self.mean, self.wave = self.init_flux_wave(flux, wave, para, T, W, fix_CO=fix_CO)
        self.nf, self.nw = self.flux.shape
        print(f"centered flux: {self.nf}, wave: {self.nw}")

    def init_flux_wave(self, flux, wave, para, T, W, fix_CO):
        flux, wave = self.get_flux_in_Wrange(flux, wave, self.Ws[W])
        flux       = self.get_flux_in_Prange(flux, para, self.Ts[T], fix_CO=fix_CO)
        mean = flux.mean(0)
        return flux - mean, mean, wave

    def init_para(self, para):
        return pd.DataFrame(data=para, columns=["F","T","L","C","O"])

    def get_flux_in_Wrange(self, flux, wave, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return flux[:, start:end], wave[start:end]

    def get_flux_in_Prange(self, flux, para, Ts, fix_CO=True):
        dfpara = self.init_para(para)
        if fix_CO:
            dfpara = dfpara[(dfpara["C"] == 0.0) & (dfpara["O"] == 0.0)]
            print(f"CO==0: {dfpara.size}")
        self.dfpara = dfpara[(dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1])]
        return flux[self.dfpara.index]
################################ RPCA #####################################
    def prepare_model(self, T, W):
        pass

    def init_rpca(self, T, W, prll):
        print("=====Initializing RPCA ======")
        G = T + W + str(int(fix_CO))
        gs_max = self.Gs[G][0]
        gl_max = self.Gs[G][1]

        self.gs = self.ratio * gs_max
        self.gl = self.ratio * gl_max
        # self.gl = 6

        abs_tol   = 1e-4 * np.sqrt(self.m * self.n * self.N)
        rel_tol   = 1e-2
        rel_tol_N = 1e-2 * np.sqrt(self.N)
        self.tol  = [abs_tol, rel_tol, rel_tol_N]
        self.rho = 1.0 / self.la

        if prll: self.pool = ThreadPool(processes=self.N) # Create thread pool for asynchronous processing


    def init_pcp(self):
        R = np.zeros((self.m, self.n))    #Residual
        S = np.zeros((self.m, self.n))    #Sparse components
        L = np.zeros((self.m, self.n))    #Low rank components
        U = np.zeros((self.m, self.n))
        z = np.zeros((self.m, self.n * self.N))
        print("\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s" %('iter', 'res', 'eps r', 
                                                  'dz', 'eps U', 'loss'))

        h = {}
        h['loss'] = np.zeros(self.epoch)
        h['res'] = np.zeros(self.epoch)
        h['dz'] = np.zeros(self.epoch)
        h['eps_r']  = np.zeros(self.epoch)
        h['eps_U'] = np.zeros(self.epoch)
        self.h = h
        return R, S, L, U, z


    def update_R(self, R, B=None):
        return (R - B) / (1.0 + self.la)

    def update_S(self, S, B=None):
        self.get_S_ratio(S)
        return RPCA.prox_l1 (S - B, self.la * self.gs)

    def get_S_ratio(self, S, out=0):
        SSum = np.sum(np.abs(S),  axis=0)
        self.S_l1 = np.sum(SSum)
        SMaxs =SSum[SSum  >  self.clip * np.max(SSum)]
        print(f"S_{len(SMaxs)}", end=" ")
        if out: return SSum, SMaxs


    def update_L(self, L, B=None):
        return self.prox_mat(L - B, self.la * self.gl)

    def prox_l1(self, mat, cutoff):
        # return np.maximum(0, S - cutoff) 
        clipped = np.maximum(0, mat - cutoff) - np.maximum(0, -mat - cutoff)
        self.get_S_ratio(clipped)
        return clipped

    def prox_mat(self, mat, r):
        u, s, vt = svds(mat, k=40, tol=1e-6)
        u, s, v = u[:,::-1], s[::-1], vt[::-1, :]
        # u, s, v = svd(mat, full_matrices=False)
        prox_s = np.maximum(s - r, 0.0) 
        self.L_eigs = prox_s[prox_s > 0.0]
        self.L_rk = len(self.L_eigs)
        u, s, v = u[:,:self.L_rk], prox_s[:self.L_rk], v[:self.L_rk, :]
        print(f"L_{self.L_rk}", end="") 
        return (u * prox_s).dot(v)

    def loss(self, R, S, L):
        noise   = norm(R, ord='fro') ** 2    # squared frobenius norm (makes X_i small)
        sparse  = self.gs * self.S_l1 # L1 norm (makes X_i sparse)
        lowrank = self.gl * self.L_eigs.sum() # nuclear norm
        # lowrank = self.gl * norm(L, ord="nuc")         # nuclear norm (makes X_i low rank)
        return noise + sparse + lowrank


    def episode(self, ep, *args):
        R, S, L, U, z = args
        # print(R.shape, S.shape, L.shape, U.shape)
        R, S, L, U = self.update_RSLU(R, S, L, U)
        RSL = np.hstack((R, S, L))
        E = (self.flux - R - S - L) / 3.0
        new_z = RSL + np.tile(E, (1, 3))

        self.h['loss'][ep]  = self.loss(R, S, L)
        self.h['res'][ep]   = (3.0 * norm(E, 'fro')**2)**0.5
        self.h['dz'][ep]    = norm(self.rho * (new_z - z), 'fro')
        self.h['eps_r'][ep] = self.tol[0] + self.tol[1] * np.maximum(norm(RSL, 'fro'), norm(new_z, 'fro'))
        self.h['eps_U'][ep] = self.tol[0] + self.tol[2] * norm(self.rho * U, 'fro')

        return R, S, L, U, new_z

    def stop(self, ep):
        if self.L_rk <=2: return True
        small_res = (self.h['res'][ep] < self.h['eps_r'][ep])
        small_dz  = (self.h['dz'][ep] < self.h['eps_U'][ep])
        return small_res and small_dz
        
    def pcp(self):
        start = time.time()
        args = self.init_pcp()
        print("\n=====Starting RPCA=====")
        for ep in range(self.epoch):
            print(f"|EP{ep+1}_", end="")
            args = self.episode(ep, *args)
            if (ep == 0) or (np.mod(ep + 1,10) == 0):
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


    def finish(self, *args):
        self.pool.close()
        self.R = args[0]
        self.S = args[1]
        self.L = args[2]
        
        self.SSum = np.mean(np.abs(self.S), axis=0)
        self.SMax = np.max(self.SSum)
        self.SClp = np.clip(self.SSum, self.SMax * self.clp, self.SMax)

    def update_RSLU(self, R, S, L, U):
        B = (R + S + L - self.flux) / 3.0 + U
        if self.pool is not None:
            R, S, L = self.update_RSL_parallel(R, S, L, B, self.pool)
        else:
            R = self.update_R(R, B)
            S = self.update_S(S, B)
            L = self.update_L(L, B)
        return R, S, L, B

    def update_RSL_parallel(self, R, S, L, B, pool): 
        async_R = pool.apply_async(lambda x: self.update_R(x, B), [R])
        async_S = pool.apply_async(lambda x: self.update_S(x, B), [S])
        async_L = pool.apply_async(lambda x: self.update_L(x, B), [L])

        R = async_R.get()
        S = async_S.get()
        L = async_L.get()
        return R, S, L
    
    def plot_S(self, ax=None):
        if ax is None: ax = plt.gca()
        ax.plot(self.wave, np.mean(np.abs(self.S), axis=0), c="r")

    def eval_pcp(self, roff=3000, soff=2000):
        res = self.R + self.S + self.L - self.flux
        plt.plot(self.wave, -res.sum(0)-roff, label=f"Res - {roff}", c='g')
        plt.plot(self.wave, abs(self.L).sum(0), label = "LowRank", c='skyblue')
        plt.plot(self.wave, self.S.sum(0)-soff, label=f"Sparse - {soff}", c='r')
        plt.xlabel("wave")
        plt.ylabel("norm flux")
        plt.title(f"Spec {self.m}  @ LowT 3500 -6500K")
        plt.legend()

    def save(self, PATH):
        print("=================SAVINHG==================")
        with h5py.File(PATH, 'w') as f:
            f.create_dataset('R', data = self.R)
            f.create_dataset('S', data = self.S)
            f.create_dataset('L', data = self.L)

    # @staticmethod
    # def update_R(x, b, la):
    #     return (1.0 / (1.0 + la)) * (x - b)
    # @staticmethod
    # def update_S(x, b, l, g, pl):
    #     return pl(x - b, l * g)
    # @staticmethod
    # def update_L(x, b, l, g, pl, pm):
    #     return pm(x - b, l * g, pl)
    # @staticmethod
    # def update_RSL(func, item):
    #     return map(func, [item])[0]

 
    @staticmethod
    def avg(*args):
        return np.mean([*args], axis=0)

