import sys
import time
import numpy as np
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool

def prox_l1(vec, cutoff):
    return np.maximum(0, vec - cutoff) - np.maximum(0, -vec - cutoff)

def prox_matrix(mat, cutoff, prox_f):
    u, s, v = svd(mat, full_matrices=False)
    prox_s = prox_f(s[:, None], cutoff)
    return u.dot(np.diagflat(prox_s)).dot(v.T)

def avg(*args):
    return np.mean([*args], axis=0)

def loss(R, gs, S, gl, L):
    noise   =      norm(R, ord='fro') ** 2  # squared frobenius norm (makes X_i small)
    sparse  = gs * norm(S.reshape(-1), ord=1)           # L1 norm (makes X_i sparse)
    lowrank = gl * norm(L, ord="nuc")       # nuclear norm (makes X_i low rank)

    return noise + sparse + lowrank


def rpcaADMM(data):
    """
    ADMM implementation of matrix decomposition. In this case, RPCA.

    Adapted from: http://web.stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html
    """

    pool = ThreadPool(processes=3) # Create thread pool for asynchronous processing

    N = 3         # the number of matrices to split into 
                  # (and cost function expresses how you want them)
 
    X = np.float_(data)    # A = S + L + V
    m, n = X.shape

    gs_max = np.sum(X)      #maximum sparsity constraint
    gl_max = norm(X, ord=2) #maximum lowrank constraint
    gs = 0.15 * gs_max
    gl = 0.15 * gl_max

    MAX_ITER = 100
    ABSTOL   = 1e-4 * np.sqrt(m * n * N)
    RELTOL   = 1e-2
    RELTOLN  = 1e-2 * np.sqrt(N)

    start = time.time()

    lambdap = 1.0
    rho = 1.0 / lambdap

    R = np.zeros((m, n))    #Residual
    S = np.zeros((m, n))    #Sparse components
    L = np.zeros((m, n))    #Low rank components
    z = np.zeros((m, N * n))
    U = np.zeros((m, n))

    print("\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s" %('iter', 'r norm', 'eps pri', 
                                                  's norm', 'eps dual', 'objective'))

    # Saving state
    h = {}
    h['loss'] = np.zeros(MAX_ITER)
    h['r_norm'] = np.zeros(MAX_ITER)
    h['s_norm'] = np.zeros(MAX_ITER)
    h['eps_pri'] = np.zeros(MAX_ITER)
    h['eps_dual'] = np.zeros(MAX_ITER)

    def update_R(x, b, l):
        return (1.0 / (1.0 + l)) * (x - b)
    def update_S(x, b, l, g, pl):
        return pl(x - b, l * g)
    def update_L(x, b, l, g, pl, pm):
        return pm(x - b, l * g, pl)

    def update(func, item):
        return map(func, [item])[0]

    for k in range(MAX_ITER):

        B = avg(R, S, L) - X / N + U

        # Original MATLAB x-update
        # X_1 = (1.0/(1.0+lambdap))*(X_1 - B)
        # X_2 = prox_l1(X_2 - B, lambdap*g2)
        # X_3 = prox_matrix(X_3 - B, lambdap*g3, prox_l1)

        # Parallel x-update
        async_R = pool.apply_async(update, (lambda x: update_R(x, B, lambdap), R))
        async_S = pool.apply_async(update, (lambda x: update_S(x, B, lambdap, gs, prox_l1), S))
        async_L = pool.apply_async(update, (lambda x: update_L(x, B, lambdap, gl, prox_l1, prox_matrix), L))

        R = async_R.get()
        S = async_S.get()
        L = async_L.get()

        # (for termination checks only)
        old_z = z
        stack = np.hstack((R, S, L))
        diff = np.tile(X /N - avg(R, S, L), (1, N))
        z = stack + diff

        # u-update
        U = B

        # diagnostics, reporting, termination checks
        h['loss'][k]   = loss(R, gs, S, gl, L)
        h['r_norm'][k]   = norm(diff, 'fro')
        h['s_norm'][k]   = norm(rho * (z - old_z), 'fro')
        h['eps_pri'][k]  = ABSTOL + RELTOL  * np.maximum(norm(stack, 'fro'), norm(z, 'fro'))
        h['eps_dual'][k] = ABSTOL + RELTOLN * norm(rho * U, 'fro')

        if (k == 0) or (np.mod(k + 1,10) == 0):
            print("%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f" %(k + 1,
                                                                  h['r_norm'][k], 
                                                                  h['eps_pri'][k], 
                                                                  h['s_norm'][k], 
                                                                  h['eps_dual'][k], 
                                                                  h['objval'][k]))
        if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
            break

    h['addm_toc'] = time.time() - start
    h['admm_iter'] = k
    h['R_admm'] = R
    h['S_admm'] = S
    h['L_admm'] = L

    return h