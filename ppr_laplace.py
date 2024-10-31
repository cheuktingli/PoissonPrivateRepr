import numpy as np
import scipy as sp
from scipy.special import gamma
import scipy.stats as sp_stats
import heapq
import matplotlib
import matplotlib.pyplot as plt
import time
from math import ceil, floor, log2
import random
from ppr import *


# The following code is for applying the 'Poisson private representation' on Laplace mechanism and metric DP.
# It will recover the Figure 2 in [1]. Discussions can be found in [1], Appendix J.
#
# [1] Universal Exact Compression of Differentially Private Mechanisms, accepted at NeurIPS 2024.
# An arxiv preprint can be found by: https://arxiv.org/pdf/2405.20782

# If you use this codebase or any part of it for a publication, please cite:
# @inproceedings{NEURIPS2024_universal,
#  author = {Liu, Yanxiao and Chen, Wei-Ning and {\"O}zg{\"u}r, Ayfer and Li, Cheuk Ting},
#  booktitle = {accepted at Advances in Neural Information Processing Systems},
#  title={Universal Exact Compression of Differentially Private Mechanisms},
#  year = {2024}
# }

def init_rng(rng):
    return np.random.default_rng(rng)


# Binary search
def bsearch(f, y, a, b, n=60):
    for _ in range(n):
        m = (a + b) * 0.5
        if f(m) > y:
            b = m
        else:
            a = m
    return (a + b) * 0.5


# n-dimensional Laplace random vector
def laplace_nd(n, eps):
    x = np.random.normal(size = n)
    x /= np.linalg.norm(x)
    g = np.random.gamma(n, 1.0/eps)
    return x * g


# Uniform point over n-ball
def random_nball(n):
    x = np.random.normal(size = n)
    x /= np.linalg.norm(x)
    r = np.random.rand() ** (1 / n)
    return x * r


# Volume of unit n-ball
def nball_vol(n):
    return np.pi ** (n/2) / sp.special.gamma(n/2 + 1)


# Log volume of unit n-ball
def nball_logvol(n):
    return np.log(np.pi) * (n/2) - sp.special.loggamma(n/2 + 1)


# Number of bits needed to quantize at resolution u
def discrete_laplace_nbits(n, u, ball=True):
    if ball:
        return int(ceil((nball_logvol(n) - np.log(u) * n) * np.log2(np.e)))
    else:
        return n * int(ceil(np.log2(2.0 / u)))


# MSE of Laplace mechanism
def laplace_mse(n, eps):
    return n * (n+1) / eps**2


# Estimate MSE of discrete Laplace
def discrete_laplace_mse(n, eps, u, discrete=True, ball=True, ntrial=10000):
    mses = []
    for _ in range(ntrial):
        if ball:
            x = random_nball(n)
        else:
            x = np.random.random(size=n) * 2 - 1

        z = x + laplace_nd(n, eps)

        # Truncate
        if ball:
            normz = np.linalg.norm(z)
            if normz > 1:
                z /= normz
        else:
            z = np.minimum(np.maximum(z, -1.0), 1.0)
        
        # Quantize
        if discrete:
            z = (np.floor(z / u) + 0.5) * u

        mse = np.linalg.norm(x - z) ** 2
        mses.append(mse)

    return (np.mean(mses), np.std(mses, ddof=1))


# Number of bits needed to for PPR-compressed Laplace mechanism
def ppr_laplace_nbits(n, eps, a):
    eta = np.log2(3.56) / min((a - 1) / 2, 1)
    ell = ((n/2) * np.log2((2/np.e) * (eps**2/n + n + 1)) 
           - (sp.special.loggamma(n + 1) - sp.special.loggamma(n/2 + 1)) * np.log2(np.e) + eta)
    return ell + np.log2(ell + 1) + 2
    

if __name__ == "__main__":
    np.random.seed(2024)

    n = 500  # number of users
    C = 10000
    a = 2.0
    print("For PPR, we set alpha = ", a)
    data_pt_size = 300
    ball = True
    ntrial = 5000

    color_list = ['red', 'green', 'blue', 'orange', 'cyan', 'black']

    ppr_mse_list = []
    discrete_mse_list = []

    eps_target_list = np.logspace(2, 5, data_pt_size)
    commu_budget_list = np.array([1, 2, 3]) * n

    plt.figure(figsize=(10, 6.5))

    max_discrete_mse_std = 0.0
    max_coeff_var = 0.0

    for i, commu_budget in enumerate(commu_budget_list):
        if ball:
            discrete_u = bsearch(lambda u: discrete_laplace_nbits(n, u), commu_budget, 10.0, 1e-6, 100)
        else:
            discrete_u = 2.0 ** (1 - commu_budget // n)

        if ball:
            ppr_eps = bsearch(lambda eps: ppr_laplace_nbits(n, eps, a), commu_budget, 1e-6, 10000.0, 100)
        else:
            ppr_eps = bsearch(lambda eps: ppr_laplace_nbits(n, eps*np.sqrt(n), a), commu_budget, 1e-6, 10000.0, 100)

        print(commu_budget, discrete_u, ppr_eps)
        pass

        ppr_mses = []
        discrete_mses = []

        for j in range(len(eps_target_list)):
            eps = eps_target_list[j]

            ppr_mses.append(laplace_mse(n, min(eps, ppr_eps)) * C**2)

            discrete_mse, discrete_mse_std = discrete_laplace_mse(n, eps, discrete_u, ball=ball, ntrial=ntrial)
            discrete_mse *= C**2
            discrete_mse_std *= C**2
            discrete_mses.append(discrete_mse)
            max_discrete_mse_std = max(max_discrete_mse_std, discrete_mse_std)
            max_coeff_var = max(max_coeff_var, discrete_mse_std / np.sqrt(ntrial) / discrete_mse)
            
        discrete_mse = np.array(discrete_mse)
        discrete_mse_std = np.array(discrete_mse_std)

        plt.loglog(eps_target_list*a*2/C, ppr_mses, '-', label=f'PPR ({commu_budget} bits)', color=color_list[i], zorder=10-i)
        plt.loglog(eps_target_list/C, discrete_mses, '--', label=f'Discrete Laplace ({commu_budget} bits)', color=color_list[i], zorder=10-i)

    print('Max coefficient of variation of mean =', max_coeff_var)
    plt.title(f'Metric privacy, dimension d = {n}')
    plt.ylabel('MSE')
    plt.xlabel('Privacy ($\epsilon$)')
    plt.legend()
    plt.xlim((1e-1, 1e1))
    plt.ylim((1e6, 1e9))
    plt.grid()
    plt.savefig(r"experiment_laplace.png", format="png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
