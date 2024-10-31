# Universal Exact Compression of Differentially Private Mechanisms

This repository includes the source code for the 'Poisson private representation' (PPR) algorithm in [Paper](https://arxiv.org/pdf/2405.20782). 

```
Universal Exact Compression of Differentially Private Mechanisms
Yanxiao Liu, Wei-Ning Chen, Ayfer Özgür, Cheuk Ting Li
accepted at Advances in Neural Information Processing Systems (NeurIPS) 2024 
```

The source code is released under the MIT License. See the License file for details.

## Requirements

To install requirements:

```setup
pip install numpy
pip install scipy
pip install matplotlib
```

# Introduction
The algorithm PPR is to compress the communication in differentially private mechanisms. The main advantages are: 

- Universality: PPR can simulate and compress **any** local or central DP mechanism; 
- Exactness: PPR ensures that the reproduced distribution **perfectly** matches the original one; 
- Communication efficiency: PPR compresses any DP mechanism to a **near-optimal** size. 

The implementation utilizes a reparametrization method to guarantee that the PPR terminates in a finite amount of time. 



# Source Code Explanations
- `ppr.py`: The PPR algorithm, as proposed in Algorithm 1 in the paper. With inputs: privacy parameter $\alpha$, reference distribution $Q$, ratio $r(z) := \frac{dP}{dQ}(z)$, bound $r^* \geq \sup_z r(z)$ and pseudorandom number generator, PPR outputs an index $k$, which is a positive integer. 

- `ppr_gaussian.py`: Evaluate PPR on Gaussian mechanisms and distributed mean estimation task. We compare it to the method in [1], and Figure 1 in our paper can be reproduced. 

- `ppr_laplace.py`: Evaluate PPR on Laplace mechanisms and metric DP. We compare it to the discrete Laplace mechanism in [2], and Figure 2 in our paper can be reproduced. 


[1] Wei-Ning Chen, Dan Song, Ayfer Özgür, and Peter Kairouz. Privacy amplification via com pression: Achieving the optimal privacy-accuracy-communication trade-off in distributed mean estimation. Advances in Neural Information Processing Systems, 36, 2023.

[2] Miguel E Andrés, Nicolás E Bordenabe, Konstantinos Chatzikokolakis, and Catuscia Palamidessi. Geo-indistinguishability: Differential privacy for location-based systems. In Proceedings of the 2013 ACM SIGSAC conference on Computer & communications security, pages 901–914, 2013. 


# Citation
If you use this codebase or any part of it for a publication, please cite:

```
@inproceedings{NEURIPS2024_universal,
 author = {Liu, Yanxiao and Chen, Wei-Ning and {\"O}zg{\"u}r, Ayfer and Li, Cheuk Ting},
 booktitle = {accepted at Advances in Neural Information Processing Systems},
 title={Universal Exact Compression of Differentially Private Mechanisms},
 year = {2024}
}
```