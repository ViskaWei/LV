# -*- coding: utf-8 -*-
"""

An implementation of the Principal Component Pursuit algorithm for robust PCA
as described in `Candes, Li, Ma, & Wright <http://arxiv.org/abs/0912.3599>`_.

An alternative Python implementation using non-standard dependencies and
different hyperparameter choices is available at:

http://blog.shriphani.com/2013/12/18/
    robust-principal-component-pursuit-background-matrix-recovery/

"""

# from __future__ import division, print_function

# __all__ = ["pcp"]

import time
# import fbpca
import logging
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


# class PCP(object):
#     def __init__(self, ds):

# ############ RPCA ###############################
#         self.X = ds.flux
#         # self.Mean = None
#         self.N = 3
#         self.Gs = {"HH0": [1.5, 260.,  1200.], 
#                    "HH1": [1.4, 260.,  270.], 
#                    "LL0": [9., 7200., 6300.], 
#                    "LL1": [8., 6000., 1400.],
#                    "LH0": [2.3, None, 860.],
#                    "LH1": [2., None, 155.],
#                    "HL0": [4.5, None, 3600.],
#                    "HL1": [4.3, None, 790],


#                     "LH": 1.14, "LL": 1.14, "TT0": [1.5,200]} 
#         self.GCs = {"LL0": [7.6, 5800., 4000.],
#                    "LH0": [2.3, None, 860.],

#                     "HH1": [185.9, 115.0] }
#         self.g = ""

#         self.gs = None
#         self.gl = None
#         self.rate = None
#         self.mask = None
#         self.la = None
#         self.tol = None
#         self.rho = None

#         self.svd_tr = None
#         self.epoch = None
#         self.pool = None        

#         self.R = None
        
#         self.L = None
#         self.L_eigs = None
#         self.L_rk = None
#         self.L_lb = 10

#         self.S = None
#         self.S_lb = 10
#         self.S_l1 = None
#         self.SSum = None
#         self.SMax = None
#         self.SClp = None
#         self.Snum = None
#         self.clp = 0.2
#         self.norm = None

#         self.cmap="YlGnBu"


# ################################ Model #####################################
#     def prepare_model(self, X=None,  rs=1.0, rl=1.0, tol=0.1, svd_tr=50, ep=100):
#         self.Xmean = self.mean if X is None else self.X.mean() 
#         self.norm  = np.sum(self.X ** 2)
#         self.svd_tr = svd_tr
#         self.epoch = ep 
#         self.tol = tol
#         self.rs = rs
#         self.rl = rl
#         print(f"lambda {self.tol} | ep {self.epoch} | svd {self.svd_tr} | rs {self.rs:.2f} | rl {self.rl:.2f} |")

# ################################ RPCA #####################################

#     def init_pcp(self):
#         self.isStop= False
#         u, w, v = self._svd(self.X, rank=10)

#         self.gl = w[-1]
        
#         self.mask = self.get_mask(v[:5], ratio=0.1)
        
#         S = self.get_S_from_mask(self.X, self.mask)
#         # L = self.flux[:, ~mask]
#         # L = np.zeros(self.size)
#         # R = np.random.uniform(0.0, self.Xmean, self.shape )
#         R = np.zeros((self.nf, self.nw))
#         return R, S


def pcp(M, delta=1e-6, mu=None, lam = None, S=None,  maxiter=500, verbose=False):
    shape = M.shape
    prod = np.prod(shape)
    # Initialize the tuning parameters.
    lam = lam or (1.0 / np.sqrt(np.max(shape)))
    if mu is None:
        mu = 0.25 * prod / np.sum(np.abs(M))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = np.sum(M ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)

    S = np.zeros(shape)
    B = np.zeros(shape)
    while i < max(maxiter, 1):
        # SVD step.
        strt = time.time()
        
        u, s, v = _svd(M - S + B, rank+1, tol=1./mu)
        # print("s", s[:20])
        svd_time = time.time() - strt

        s = shrink(s, 1./mu)
        rank = np.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = (u * s).dot(v)

        # Shrinkage step.
        S = shrink(M - L + B, lam / mu)
        # S = np.maximum(M - L + B - (lam / mu), 0.0)
        # Lagrange step.
        R = M - L - S
        B += R

        # Check for convergence.
        err = np.sqrt(np.sum(R ** 2) / norm)
        if verbose and ((i==0) or (i % 5 == 0)):
            print(f"EP{i}_e{err:.2e}_L{rank:d}_S{(np.sum(S>0) / prod):.2f}", end=" ")
        if err < delta:
            break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return L, S, (u, s, v)

# @staticmethod
def shrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), 0.0)

# @staticmethod
def _svd(X, rank, tol=1e-5):
    rank = min(rank, 500)
    u, s, v = svds(X, k=rank, tol=tol)
    u, s, v = u[:, ::-1], s[::-1], v[::-1, :]
    return u, s, v


def plot_LS(S, v, wave, ax=None):
    ax = ax or plt.gca()
    plt.plot(wave, np.mean(np.abs(S), axis=0), c="k")

    for i in range(min(len(v),5)):
        plt.plot(wave, v[i] + 0.3*(1 + i))


from matplotlib.colors import LogNorm

def plot_pcp(M, L, S, u, s, v, wave, cmap="hot"):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(M), aspect="auto", cmap=cmap,)
    plt.title("Original Flux")
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(L), aspect="auto", cmap=cmap)
    plt.title("Low rank matrix")
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(S), cmap=cmap, aspect="auto", )
    plt.title("Sparse matrix")
    plt.subplot(2, 2, 4)
    for i in range(min(len(v),5)):
        plt.plot(wave, v[i] + 0.3*(1 + i))
    plt.plot(wave, np.mean(np.abs(S), axis=0), c="k")
    # plt.imshow(np.dot(u, np.dot(np.diag(s), v)), cmap="gray")
    plt.title("L & S")
    plt.show()