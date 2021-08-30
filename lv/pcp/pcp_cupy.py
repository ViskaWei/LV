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

import time
import logging
import cupy as cp
import numpy as np


def pcp_cupy(M, delta=1e-6, mu=None, lam=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", **svd_args):
    # Check the SVD method.
    allowed_methods = ["cupy"]
    
    M = cp.asarray(M) 
    
    if svd_method not in allowed_methods:
        raise ValueError("'svd_method' must be one of: {0}"
                         .format(allowed_methods))

    # Check for missing data.
    shape = M.shape
    if missing_data:
        missing = ~(cp.isfinite(M))
        if cp.any(missing):
            M = cp.array(M)
            M[missing] = 0.0
    else:
        missing = cp.zeros_like(M, dtype=bool)
        if not cp.all(cp.isfinite(M)):
            logging.warn("The matrix has non-finite entries. "
                         "SVD will probably fail.")

    # Initialize the tuning parameters.
    if lam is None:
        lam = 1.0 / cp.sqrt(cp.max(shape))
    if mu is None:
        mu = 0.25 * cp.prod(shape) / cp.sum(cp.abs(M))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = cp.sum(M ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)
    S = cp.zeros(shape)
    Y = cp.zeros(shape)
    while i < max(maxiter, 1):
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, M - S + Y / mu, rank+1, 1./mu, **svd_args)
        svd_time = time.time() - strt

        s = shrink(s, 1./mu)
        rank = cp.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = cp.dot(u, cp.dot(cp.diag(s), v))

        # Shrinkage step.
        S = shrink(M - L + Y / mu, lam / mu)

        # Lagrange step.
        step = M - L - S
        step[missing] = 0.0
        Y += mu * step

        # Check for convergence.
        err = cp.sqrt(cp.sum(step ** 2) / norm)
        if verbose:
            print("Not impemented")
            # we do not want to move arrays back and forth between CPU and GPU
            #print(("Iteration {0}: error={1:.3e}, rank={2:d}, nnz={3:d}, "
            #       "time={4:.3e}")
            #      .format(i, err, cp.sum(s > 0), cp.sum(S > 0), svd_time))
        if err < delta:
            break
        i += 1

    if (i >= maxiter) and verbose:
        logging.warn("convergence not reached in pcp")
    return cp.asnumpy(L), cp.asnumpy(S), (cp.asnumpy(u), cp.asnumpy(s), cp.asnumpy(v))


def shrink(M, tau):
    sgn = cp.sign(M)
    S = cp.abs(M) - tau
    S[S < 0.0] = 0.0
    return sgn * S


def _svd(method, X, rank, tol, **args):
    # rank = min(rank, np.min(X.shape))
    rank = min(rank, 100)

    if method == "cupy":
        return cp.linalg.svd(X, full_matrices=False, **args)
    elif method == "sparse":
        if rank >= np.min(X.shape):
            return cp.linalg.svd(X, full_matrices=False)
        u, s, v = svds(X, k=rank, tol=tol)
        u, s, v = u[:, ::-1], s[::-1], v[::-1, :]
        return u, s, v
    raise ValueError("invalid SVD method")
