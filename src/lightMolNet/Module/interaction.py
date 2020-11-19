# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : interaction.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from functools import partial

import torch
from lightMolNet.Module.activations import shifted_softplus
from lightMolNet.Module.convolution import CFConv
from lightMolNet.Module.cutoff import CosineCutoff
from lightMolNet.Module.util import Dense
from torch import nn

shifted_softplus = partial(shifted_softplus, shift=2)


class SimpleAtomInteraction(nn.Module):
    r"""simple interaction layers for modeling interactions of atomistic systems.

    Args
    ----
        n_atom_embeddings:int
            number of features to describe atomic environments.
        n_spatial_basis:int
            number of input features of filter-generating networks.
        n_filters:int
            number of filters used in continuous-filter convolution.
        cutoff:float
            cutoff radius
        cutoff_network:nn.Module, optional
            cutoff layer
        normalize_filter:bool, optional
            if True, divide aggregated filter by number of neighbors over which convolution is applied.

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
            n_atom_embeddings: int,
            n_spatial_basis: int,
            n_filters: int,
            cutoff: float,
            cutoff_network: nn.Module = CosineCutoff,
            normalize_filter: bool = False,
    ):
        super(SimpleAtomInteraction, self).__init__()
        self.filter_network = nn.Sequential(
            Dense(
                n_spatial_basis,
                n_filters,
                activation=shifted_softplus
            ),
            Dense(
                n_filters,
                n_filters
            ),
        )

        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)

        # interaction convolutional layers
        self.cfconv = CFConv(
            n_atom_embeddings,
            n_filters,
            n_atom_embeddings,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )

        # dense layer
        self.dense = Dense(
            n_atom_embeddings,
            n_atom_embeddings,
            bias=True,
            activation=None)

    def forward(
            self,
            x: torch.Tensor,
            r_ij: torch.Tensor,
            neighbors: torch.Tensor,
            neighbor_mask: torch.Tensor,
            f_ij: torch.Tensor = None
    ):
        r"""Compute interaction output.

        Parameters
        ----------
            x:torch.Tensor
                input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij:torch.Tensor
                interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors:torch.Tensor
                indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask:torch.Tensor
                mask to filter out non-existing neighbors
                introduced via padding.
            f_ij:torch.Tensor, optional
                expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns
        -------
            v:torch.Tensor
                block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v
