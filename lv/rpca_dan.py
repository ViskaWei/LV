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



def pcp(M, delta=1e-6, mu=None, lam = None, S=None,  maxiter=500, verbose=False):
    shape = M.shape
    prod = np.prod(shape)
    # Initialize the tuning parameters.
    lam = lam or 1.0 / np.sqrt(np.max(shape))
    if mu is None:
        mu = 0.25 * prod / np.sum(np.abs(M))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = np.sum(M ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)
    u, s, v = _svd(M, 40, tol=1e-2)
    vv = v[0] 
    cut = np.quantile(vv, 0.2)
    vv[vv < cut] = 0
    S = np.tile(vv, (shape[0], 1))
    # S = 
    Y = np.zeros(shape)
    while i < max(maxiter, 1):
        # SVD step.
        # strt = time.time()
        # u, s, v = _svd(M - S + Y / mu, rank+1, 1./mu)
        B = Y / mu
        # B = step
        u, s, v = _svd(M - S + B, rank+1)

        # svd_time = time.time() - strt
        s = shrink(s, 1./mu)
        # s = np.maximum(s - 1./mu, 0.0, out=s)
        rank = np.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = np.dot(u, np.dot(np.diag(s), v))
        # L = (u * s).dot(u)

        # Shrinkage step.
        S = shrink(M - L + B, 1 / (lam * mu))

        # Lagrange step.
        R = M - L - S
        Y += mu * R

        # Check for convergence.
        err = np.sqrt(np.sum(R ** 2) / norm)
        if verbose:
            print(("Iteration {0}: error={1:.3e}, rank={2:d}, nnz={3:.4f}")
                  .format(i, err, np.sum(s > 0), np.sum(S > 0)/prod))
        if err < delta:
            break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return L, S, (u, s, v)


def shrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), 0.0)
    # sgn = np.sign(M)
    # S = np.abs(M) - tau
    # S[S < 0.0] = 0.0
    # return sgn * S


def _svd(X, rank, tol=1e-2):
    rank = min(rank, 50)
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
    plt.imshow(np.abs(M), cmap=cmap,)
    plt.title("Original Flux")
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(L), cmap=cmap)
    plt.title("Low rank matrix")
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(S), cmap=cmap, )
    plt.title("Sparse matrix")
    plt.subplot(2, 2, 4)
    for i in range(min(len(v),5)):
        plt.plot(wave, v[i] + 0.3*(1 + i))
    plt.plot(wave, np.mean(np.abs(S), axis=0), c="k")
    # plt.imshow(np.dot(u, np.dot(np.diag(s), v)), cmap="gray")
    plt.title("L & S")
    plt.show()