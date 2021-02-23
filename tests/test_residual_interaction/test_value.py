# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : test_value.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch

from lightMolNet.Module.cutoff import RBFCutoff
from lightMolNet.Module.interaction import AtomInteractionWithResidual
from lightMolNet.Module.neighbors import AtomDistances


class Test_RbfShape:
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
        rbflayer = RBFCutoff(K, 10)
        emb = torch.nn.Embedding(10, 128, padding_idx=0)
        x = emb(torch.LongTensor([[1, 6, 6, 8]]))
        InteractionLayer = AtomInteractionWithResidual(128, K, 0, 0)
        rbf_value = rbflayer(Dij)
        v = InteractionLayer(x, rbf_value, Nbh)
        print(x.shape, v.shape)
