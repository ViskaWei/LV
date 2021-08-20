import sys
import time
import numpy as np
from tqdm import tqdm
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool


class RPCA(object):
    def __init__(self, data, la=1.0,  n_iter=100, ratio=0.15, parallel=0):
        self.X = data
        self.m, self.n = self.X.shape
        self.la = la
        self.n_iter = n_iter
        self.N = 3
        self.tol = None
        self.gs = None
        self.gl = None
        self.pool = None
        self.rho = None
        self.R = None
        self.S = None
        self.L = None

        self.init(data, ratio, parallel)

    def init(self, data, ratio, parallel):
        gs_max = np.abs(data).max()
        gl_max = norm(data, ord=2)
        self.gs = ratio * gs_max
        self.gl = ratio * gl_max

        abs_tol   = 1e-4 * np.sqrt(self.m * self.n * self.N)
        rel_tol   = 1e-2
        rel_tol_N = 1e-2 * np.sqrt(self.N)
        self.tol  = [abs_tol, rel_tol, rel_tol_N]
        self.rho = 1.0 / self.la

        if parallel:
            self.pool = ThreadPool(processes=self.N) # Create thread pool for asynchronous processing


    def init_pcp(self):
        R = np.zeros((self.m, self.n))    #Residual
        S = np.zeros((self.m, self.n))    #Sparse components
        L = np.zeros((self.m, self.n))    #Low rank components
        U = np.zeros((self.m, self.n))
        z = np.zeros((self.m, self.n * self.N))
        print("\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s" %('iter', 'res', 'eps r', 
                                                  'dz', 'eps U', 'loss'))

        h = {}
        h['loss'] = np.zeros(self.n_iter)
        h['res'] = np.zeros(self.n_iter)
        h['dz'] = np.zeros(self.n_iter)
        h['eps_r']  = np.zeros(self.n_iter)
        h['eps_U'] = np.zeros(self.n_iter)
        self.h = h
        return R, S, L, U, z


    def update_R(self, R, B=None):
        return (R - B) / (1.0 + self.la)

    def update_S(self, S, B=None):
        return RPCA.prox_l1 (S - B, self.la * self.gs)

    def update_L(self, L, B=None):
        return self.prox_mat(L - B, self.la * self.gl)

    def prox_mat(self, mat, r):
        u, s, v = svd(mat, full_matrices=False)
        prox_s = np.maximum(s - r, 0.0) 
        return (u * prox_s).dot(v)
        # prox_s = RPCA.prox_l1(s, cutoff)
        # if self.m >= self.n:
        #     return u.dot(np.diagflat(prox_s)).dot(v.T)
        # else:
        #     return u.dot(np.diagflat(prox_s)).dot(v)

    def loss(self, R, S, L):
        noise   =           norm(R, ord='fro') ** 2    # squared frobenius norm (makes X_i small)
        sparse  = self.gs * np.abs(S).sum() # L1 norm (makes X_i sparse)
        lowrank = self.gl * norm(L, ord="nuc")         # nuclear norm (makes X_i low rank)
        return noise + sparse + lowrank


    def episode(self, ep, *args):
        R, S, L, U, z = args
        # print(R.shape, S.shape, L.shape, U.shape)
        R, S, L, U = self.update_RSLU(R, S, L, U)
        RSL = np.hstack((R, S, L))
        E = (self.X - R - S - L) / 3.0
        new_z = RSL + np.tile(E, (1, 3))

        self.h['loss'][ep]  = self.loss(R, S, L)
        self.h['res'][ep]   = (3.0 * norm(E, 'fro')**2)**0.5
        self.h['dz'][ep]    = norm(self.rho * (new_z - z), 'fro')
        self.h['eps_r'][ep] = self.tol[0] + self.tol[1] * np.maximum(norm(RSL, 'fro'), norm(new_z, 'fro'))
        self.h['eps_U'][ep] = self.tol[0] + self.tol[2] * norm(self.rho * U, 'fro')

        return R, S, L, U, new_z

    def stop(self, ep):
        small_res = (self.h['res'][ep] < self.h['eps_r'][ep])
        small_dz  = (self.h['dz'][ep] < self.h['eps_U'][ep])
        return small_res and small_dz

        return self.TOL
    def pcp(self):
        start = time.time()
        args = self.init_pcp()
        for ep in range(self.n_iter):
            args = self.episode(ep, *args)
            if (ep == 0) or (np.mod(ep + 1,10) == 0):
                print("%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f" %(ep + 1,
                        self.h['res'][ep], self.h['eps_r'][ep], self.h['dz'][ep], 
                        self.h['eps_U'][ep], self.h['loss'][ep]))
            if self.stop(ep):
                break
        self.h["t"] = time.time() - start
        self.h["ep"] = ep
        self.R = args[0]
        self.S = args[1]
        self.L = args[2]

    def update_RSLU(self, R, S, L, U):
        B = (R + S + L - self.X) / 3.0 + U
        if self.pool is not None:
            R, S, L = self.update_RSL_parallel(R, S, L, B, self.pool)
        else:
            R = self.update_R(R, B)
            S = self.update_S(S, B)
            L = self.update_L(L, B)
        return R, S, L, B

    def update_RSL_parallel(self, R, S, L, B, pool): 
        async_R = pool.apply_async(lambda x: update_R(x, B), R)
        async_S = pool.apply_async(lambda x: update_S(x, B), S)
        async_L = pool.apply_async(lambda x: update_L(x, B), L)

        R = async_R.get()
        S = async_S.get()
        L = async_L.get()
        return R, S, L
    
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
    def prox_l1(vec, cutoff):
        # return np.maximum(0, S - cutoff) 
        return np.maximum(0, vec - cutoff) - np.maximum(0, -vec - cutoff)

    @staticmethod
    def avg(*args):
        return np.mean([*args], axis=0)

