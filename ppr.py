import numpy as np
import scipy as sp
import heapq
import random


def init_rng(rng):
    return np.random.default_rng(rng)

# The following code is for the 'Poisson private representation', a method to compress any differentially private
# algorithm, proposed in [1].
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

class PoiPrivateRepr:
    def __init__(self, encoder=True, decoder=True, Q=None, rng_common=None, rng_private=None, a=3.0):
        """ Initialize encoder/decoder
        encoder: Whether this can be used as encoder
        decoder: Whether this can be used as decoder
        Q: A function taking an RNG as input, and output a sample following distribution Q
        rng_common: RNG shared between encoder and decoder
        rng_private: Private RNG only used by this encoder/decoder
        a: Parameter alpha of PPR
        """

        self.encoder = encoder
        self.decoder = decoder
        self.Q = Q
        self.rng_common = init_rng(rng_common)
        self.rng_private = init_rng(rng_private)
        self.a = a


    def encode(self, r, r_bd):
        """ Perform encoding following Algorithm 1 in the paper [1].
        r: Function that gives the ratio dP/dQ
        r_bd: An upper-bound on the values of r
        Returns: Pair (k, z) where k is the index and z is the sample
        """

        a = self.a
        Q = self.Q
        rng_c = init_rng(self.rng_common.integers(2**60))
        rng_p = self.rng_private
        u = 0
        ws = np.inf
        k = 0
        ks = 0
        zs = 0.0
        n = 0
        g1 = sp.special.gammainc(1 - 1/a, 1) * sp.special.gamma(1 - 1/a)
        h = []

        sprob = (1/np.e) / (1/np.e + g1)

        while True:
            u += rng_p.exponential()  # Utilizing local randomness (not the PRNG)
            b = (u * a / (1/np.e + g1)) ** a
            bpia = b ** (1/a)

            if n == 0 and b * r_bd**-a >= ws:  # When no possible points are left and future points are impossible
                return (ks, zs)

            if rng_p.random() < sprob:  # Run with probability (1/e) / (1/e+g1)
                t = bpia
                v = rng_p.exponential() + 1
            else:
                v = 2
                while v > 1:
                    v = rng_p.gamma(1 - 1/a)  # Assign Gamma distribution

                t = bpia / v**(1/a)

            th = 1 if (t / r_bd) ** a * v <= ws else 0   # Check if it is possible that this point is optimal
            heapq.heappush(h, (t, v, th))
            n += th  # Number of possible points in the heap

            while h and h[0][0] <= bpia:  # Assign Zi's to points in heap with small Ti (see paper [1] for theory)
                t, v, th = heapq.heappop(h)
                n -= th
                k += 1
                z = Q(rng_c)
                w = (t / r(z)) ** a * v
                if w < ws:
                    ws = w
                    ks = k
                    zs = z

    def decode(self, k):
        """ Perform decoding following Algorithm 1 in the paper [1].
        k: index
        Returns: Sample z
        """

        Q = self.Q
        rng_c = init_rng(self.rng_common.integers(2**60))
        z = None
        for i in range(k):
            z = Q(rng_c)
        return z

