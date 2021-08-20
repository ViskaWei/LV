import sys
import time
import numpy as np
from tqdm import tqdm
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool


class RPCA(object):
    def __init__(self, data, la=1.0, ratio=0.15, parallel=0, n_iter=100):
        self.X = data
        self.N = 3
        self.parallel = parallel
        self.m, self.n = self.X.shape
        self.n_iter = n_iter
        self.tol = None
        self.gs = None
        self.gl = None
        self.la = la
        self.rho = 1.0 / self.la

        self.init(data, ratio)

    def init(self, data, ratio):
        gs_max = np.sum(data)
        gl_max = norm(data, ord=2)
        self.gs = ratio * gs_max
        self.gl = ratio * gl_max

        abs_tol   = 1e-4 * np.sqrt(self.m * self.n * self.N)
        rel_tol   = 1e-2
        rel_tol_N = 1e-2 * np.sqrt(self.N)
        self.tol  = [abs_tol, rel_tol, rel_tol_N]

    def init_pcp(self):
        R = np.zeros((self.m, self.n))    #Residual
        S = np.zeros((self.m, self.n))    #Sparse components
        L = np.zeros((self.m, self.n))    #Low rank components
        U = np.zeros((self.m, self.n))
        z = np.zeros((self.m, self.n * self.N))
        print("\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s" %('iter', 'err', 'eps r', 
                                                  'dz', 'eps U', 'loss'))

        h = {}
        h['loss'] = np.zeros(self.n_iter)
        h['err'] = np.zeros(self.n_iter)
        h['dz'] = np.zeros(self.n_iter)
        h['eps_r']  = np.zeros(self.n_iter)
        h['eps_U'] = np.zeros(self.n_iter)
        return R, S, L, U, z, h


    def update_R(self, R, B):
        return (R - B) / (1.0 + self.la)

    def update_S(self, S, B):
        return RPCA.prox_l1 (S - B, self.la * self.gs)

    def update_L(self, L, B):
        return self.prox_mat(L - B, self.la * self.gl)

    def prox_mat(self, mat, cutoff):
        u, s, v = svd(mat, full_matrices=False)
        prox_s = RPCA.prox_l1(s[:, None], cutoff)
        if self.m >= self.n:
            return u.dot(np.diagflat(prox_s)).dot(v.T)
        else:
            return u.dot(np.diagflat(prox_s)).dot(v)

    def loss(self, R, S, L):
        noise   =           norm(R, ord='fro') ** 2    # squared frobenius norm (makes X_i small)
        sparse  = self.gs * norm(S.reshape(-1), ord=1) # L1 norm (makes X_i sparse)
        lowrank = self.gl * norm(L, ord="nuc")         # nuclear norm (makes X_i low rank)
        return noise + sparse + lowrank


    def episode(self, k, *args):
        R, S, L, U, z, h = args
        # print(R.shape, S.shape, L.shape, U.shape)
        R, S, L, U = self.update_RSLU(R, S, L, U)
        RSL = np.hstack((R, S, L))
        E = (self.X - R - S - L) / 3.0
        
        new_z = RSL + np.tile(E, (1, 3))

        h['loss'][k]  = self.loss(R, S, L)
        h['err'][k]   = (3.0 * norm(E, 'fro')**2)**0.5
        h['dz'][k]    = norm(self.rho * (new_z - z), 'fro')
        h['eps_r'][k] = self.tol[0] + self.tol[1] * np.maximum(norm(RSL, 'fro'), norm(z, 'fro'))
        h['eps_U'][k] = self.tol[0] + self.tol[2] * norm(self.rho * U, 'fro')

        return R, S, L, U, new_z, h

    def stop(self):
        return (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k])

        return self.TOL
    def pcp(self):
        args = self.init_pcp()
        for ep in range(self.n_iter):
            args = self.episode(ep, *args)
            if (ep == 0) or (np.mod(ep + 1,10) == 0):
                print("%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f" %(ep + 1,
                                                                  h['err'][ep], 
                                                                  h['eps_r'][ep], 
                                                                  h['dz'][ep], 
                                                                  h['eps_U'][ep], 
                                                                  h['loss'][ep]))
            if self.stop():
                break
        return h

    def update_RSLU(self, R, S, L, U):
        B = (R + S + L - self.X) / 3.0 + U
        R = self.update_R(R, B)
        S = self.update_S(S, B)
        L = self.update_L(L, B)
        U = B
        return R, S, L, U

    
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
        return np.maximum(0, vec - cutoff) - np.maximum(0, -vec - cutoff)

    @staticmethod
    def avg(*args):
        return np.mean([*args], axis=0)

