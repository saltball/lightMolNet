# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : convolution.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import torch
from lightMolNet.Module.util import Dense, Aggregate
from torch import nn


class CFConv(nn.Module):
    r"""Continuous-filter convolution block.

        Parameters
        ----------
            n_in:int
                number of input (i.e. atomic embedding) dimensions.
            n_filters:int
                number of filter dimensions.
            n_out:int
                number of output dimensions.
            filter_network:nn.Module
                filter block.
            cutoff_network:nn.Module, optional
                if None, no cut off function is used.
            activation:callable, optional
                if None, no activation function is used.
            normalize_filter:bool, optional
                If True, normalize filter to the number of neighbors when aggregating.
            axis:int, optional
                axis over which convolution should be applied.
        References
        ----------
            .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
               Quantum-chemical insights from deep tensor neural networks.
               Nature Communications, 8, 13890. 2017.
            .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
               SchNet: A continuous-filter convolutional neural network for modeling quantum
               interactions.
               In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
            .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
               SchNet - a deep learning architecture for molceules and materials.
               The Journal of Chemical Physics 148 (24), 241722. 2018.
        """

    def __init__(
            self,
            n_in: int,
            n_filters: int,
            n_out: int,
            filter_network: nn.Module,
            cutoff_network: nn.Module = None,
            activation: callable = None,
            normalize_filter: bool = False,
            axis: int = 2,
    ):
        super(CFConv, self).__init__()
        self.in2f = Dense(
            n_in,
            n_filters,
            bias=False,
            activation=None
        )
        self.f2out = Dense(
            n_filters,
            n_out,
            bias=True,
            activation=activation
        )
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(
            axis=axis,
            mean=normalize_filter
        )

    def forward(
            self,
            x: torch.Tensor,
            r_ij: torch.Tensor,
            neighbors: torch.Tensor,
            pairwise_mask: torch.Tensor,
            f_ij: torch.Tensor = None
    ):
        r"""Compute convolution block.

        Args
        ----
            x:torch.Tensor
                input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij:torch.Tensor
                interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors:torch.Tensor
                indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask:torch.Tensor
                mask to filter out non-existing neighbors
                introduced via padding.
            f_ij:torch.Tensor, optional
                expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns
        -------
            y:torch.Tensor
                block output with (N_b, N_a, n_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W = self.filter_network(f_ij)
        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)
        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        return y
