# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : test_value.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import pytest
import torch

from lightMolNet.Module.cutoff import RBFCutoff
from lightMolNet.Module.neighbors import AtomDistances


class Test_RbfValue:
    def test_rbf_value(self):
        K = 64
        cutoff = 10
        get_dis = AtomDistances()
        R = np.array([[[0, 0, 0],
                       [1, 0, 1],
                       [-1, 0, -1],
                       [50, 50, 50]]])
        Nbh = np.array([[[1, 2, 3],
                         [0, 2, 3],
                         [0, 1, 3],
                         [0, 1, 2]]])
        R = torch.Tensor(R)
        Nbh = torch.LongTensor(Nbh)
        Dij = get_dis(R, Nbh)
        rbflayer = RBFCutoff(K, cutoff)
        rbf_value = rbflayer(Dij)
        ref_value = np.zeros([1, 4, 3, K])
        centers = np.linspace(1.0, np.exp(-cutoff), K)
        widths = (0.5 / ((1.0 - np.exp(-cutoff)) / K)) ** 2
        phi = np.zeros_like(Dij)
        for i in range(4):
            for j in range(4):
                if j == i:
                    pass
                else:
                    jj = j
                    if j > i:
                        jj = j - 1
                    for item in range(K):
                        rij = Dij[0][i][jj]
                        rr = rij / cutoff
                        phik = (1 - 6 * rr ** 5 + 15 * rr ** 4 - 10 * rr ** 3) if rr < 1 else 0
                        phi[0][i][jj] = phik
                        muk = centers[item]
                        ref_value[0][i][jj][item] = phik * np.exp(-widths * (np.exp(-rij) - muk) ** 2)

        # print(ref_value)
        print(rbf_value)
        assert rbf_value.detach().numpy() == pytest.approx(ref_value, abs=1e-6)
