# -*- coding: utf-8 -*-
"""
An implementation of the Principal Component Pursuit algorithm for robust PCA
as described in `Candes, Li, Ma, & Wright <http://arxiv.org/abs/0912.3599>`_.

An alternative Python implementation using non-standard dependencies and
different hyperparameter choices is available at:

http://blog.shriphani.com/2013/12/18/robust-principal-component-pursuit-background-matrix-recovery/

"""

from __future__ import division, print_function

__all__ = ["pcp_cupy"]

from  time import time
import logging
import cupy as cp


def pcp_cupy(M, delta=1e-6, mu=None, lam=None, norm=None, maxiter=50):    
    M = cp.asarray(M, dtype=cp.float32)
    shape = M.shape
    # Initialize the tuning parameters.
    if lam is None:
        # lam=0.003
        lam = 1.0 / cp.sqrt(shape[0])

    if mu is None:
        # mu = 11.0
        mu = 0.25 * shape[0] * shape[1] / cp.sum(cp.abs(M))
    # Convergence criterion.
    if norm is None:
        norm = cp.sum(M ** 2)
    # norm = 320862.3
    print(f"mu {mu:.2f}, lambda {lam:.4f}, norm {norm:.1f}")

    # Iterate.
    i = 0
    rank = shape[1]
    S = cp.zeros(shape)
    Y = cp.zeros(shape)
    while i < max(maxiter, 1):
        # SVD step.
        t = time()
        u, s, v = cp.linalg.svd(M - S + Y / mu, full_matrices=False)
        # u, s, v = _svd(svd_method, M - S + Y / mu, rank+1, 1./mu, **svd_args)
        svd_time = time() - t

        s = shrink(s, 1./mu)
        rank = cp.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = cp.dot(u, cp.dot(cp.diag(s), v))

        # Shrinkage step.
        S = shrink(M - L + Y / mu, lam / mu)

        # Lagrange step.
        step = M - L - S
        # step[missing] = 0.0
        Y += mu * step

        # Check for convergence.
        err = cp.sqrt(cp.sum(step ** 2) / norm)

            # we do not want to move arrays back and forth between CPU and GPU
            #print(("Iteration {0}: error={1:.3e}, rank={2:d}, nnz={3:d}, "
            #       "time={4:.3e}")
            #      .format(i, err, cp.sum(s > 0), cp.sum(S > 0), svd_time))
        if err < delta:
            break
        i += 1

    if (i >= maxiter):
        logging.warn("convergence not reached in pcp")
    else:
        print(i)
    return L, S, (u,s,v)
    # return cp.asnumpy(L), cp.asnumpy(S), (cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(v))


def shrink(M, tau):
    sgn = cp.sign(M)
    S = cp.abs(M) - tau
    S[S < 0.0] = 0.0
    return sgn * S

