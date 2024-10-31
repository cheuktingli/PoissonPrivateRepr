import numpy as np
import scipy as sp
from scipy.special import gamma
import scipy.stats as sp_stats
import heapq
import matplotlib
import matplotlib.pyplot as plt
import time
import math
import random
from ppr import *


def init_rng(rng):
    return np.random.default_rng(rng)

# The following code is for applying the Poisson private representation on distributed mean estimation.
# It will recover the Figure 1 in [1]. More discussions can be found in [1], Section 7.
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


def mixture_gaussian_log_pdf(x, local_sigma, n, bern_bias):
    logpdf1 = sp_stats.norm.logpdf(x, loc=1, scale=local_sigma)
    logpdf2 = sp_stats.norm.logpdf(x, loc=-1, scale=local_sigma)
    b = [bern_bias, 1 - bern_bias]
    return sp.special.logsumexp([logpdf1, logpdf2], b = b)


def log_integrate(logf, a, b, n_div=10000):
    xs = np.linspace(a, b, n_div * 2 + 1)
    logys = [logf(x) for x in xs]
    return sp.special.logsumexp(logys,
                                b=[(1 if i == 0 or i == n_div * 2 else 2 if i % 2 == 0 else 4) * (b - a) / (6.0 * n_div)
                                   for i in range(n_div * 2 + 1)])


def bsearch(f, y, a, b, n=60):
    for _ in range(n):
        m = (a + b) * 0.5
        if f(m) > y:
            b = m
        else:
            a = m
    return (a + b) * 0.5


if __name__ == "__main__":
    print("For the CSGM scheme we compare with, please find their codes online")
    # CSGM method in paper "Privacy amplification via compression: Achieving the optimal privacy-accuracy-communication
    # trade-off in distributed mean estimation."

    np.random.seed(2024)
    n = 500 # number of users
    a = 2  # parameter 'alpha' for PPR
    d = 1000
    d_chunk = 50 # dimension of each chunk

    delta_target = 10 ** -6

    # privacy parameter lists
    data_pt_size = 25
    eps_target_list = np.logspace(math.log10(0.05), math.log10(6),
                                  data_pt_size)

    num_itr = 50
    color_list = ['blue', 'green', 'red', 'orange', 'cyan', 'black']
    plt.figure(figsize=(12, 9))
    bern_bias = 0.5

    ppr_mse_list = []
    planar_laplace_mse_list = []

    # sigma_renyi_list is found by the Renyi DP method used in the following paper:
    # CSGM method in paper "Privacy amplification via compression: Achieving the optimal privacy-accuracy-communication
    # trade-off in distributed mean estimation."
    sigma_renyi_list =[4.882086889028869, 3.9679189741491427, 3.3364211942171096, 2.8634312157009845,
                       2.3019819419459964, 1.8747964864360256, 1.5538586528691667, 1.2877791526989313,
                       1.0672471110410697, 0.8844837768265279, 0.7330586255193339, 0.6075917051930446,
                       0.5037058084326418, 0.41768527080421336, 0.3464718975010328, 0.2875101876725239,
                       0.2387659731994063, 0.19835389211948495, 0.16497265773978143, 0.13732613588217646,
                       0.11441668029874563, 0.09548273251311912, 0.0797361920831463, 0.06675924500996189,
                       0.05615120198854129]

    mse_list = []
    size_list = [50, 100, 200, 400]
    z_order_list = [40, 30, 20, 10]

    def search_sigma(local_sigma, d):
        max_logP = np.log(1 / (np.sqrt(2 * np.pi) * local_sigma)) + 0.01

        def diff_entr(x):
            logP = mixture_gaussian_log_pdf(x, local_sigma, n, bern_bias)
            return np.log(max_logP - logP) + logP

        H_Z = log_integrate(diff_entr, -1 - 6 * local_sigma, 1 + 6 * local_sigma)
        H_Z = (np.exp(H_Z) - max_logP) * np.log2(np.e)
        H_Z_given_X = 0.5 * np.log2(2 * np.pi * np.e * (local_sigma ** 2))
        MI = H_Z - H_Z_given_X
        E_logK = MI * d + np.log2(3.56) / min((a - 1) / 2, 1)
        return E_logK + np.log2(E_logK + 1) + 2

    for i in range(len(size_list)):
        size = size_list[i]
        sigma_cutoff = bsearch(lambda sigma: search_sigma(sigma, d), size, 2000, 0.001, 30) / np.sqrt(n)
        print("Plotting size", size, "sigma_cutoff", sigma_cutoff)

        size_temp = []
        for j in range(len(eps_target_list)):
            size_temp.append((max(sigma_renyi_list[j], sigma_cutoff) ** 2))
        if size_list[i] == 400:
            lww = 3
        else:
            lww = 2.5
        plt.loglog(eps_target_list, size_temp, label=f'PPR ({size} bits)', color=color_list[i], lw=lww,
                   zorder=z_order_list[i])

    mse_uncompress_list = []
    for j in range(len(sigma_renyi_list)):
        mse_uncompress_list.append(sigma_renyi_list[j] ** 2)
    plt.loglog(eps_target_list, mse_uncompress_list, label=f'Gaussian', color='black', lw=1, zorder=100)

    chunk_mse = []
    sigma_cutoff_chunk = bsearch(lambda sigma: search_sigma(sigma, d_chunk), 400 / (d/d_chunk), 2000, 0.001, 30) / np.sqrt(n)
    for j in range(len(eps_target_list)):
        chunk_mse.append((max(sigma_renyi_list[j], sigma_cutoff_chunk) ** 2))
    plt.loglog(eps_target_list, chunk_mse, label=f'sliced PPR ($400$ bits)', color='magenta',
               linestyle='dotted', lw=3, zorder=10)

    plt.xlim([0.05, 6])  # fit the range of epsilons we care
    plt.ylim([min(mse_uncompress_list) * 0.8, max(mse_uncompress_list) * 1.2])
    plt.grid(which='major', axis='both', )
    plt.yscale('log')
    plt.ylabel('MSE', fontsize=26)
    plt.xlabel('Privacy ($\epsilon$)', fontsize=26)
    plt.title(f'Mean estimation (n = {n}, d = {d}, $\delta=1e{-6}$)', fontsize=26)
    plt.legend(fontsize=18.0, loc="lower left")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
