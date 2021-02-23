# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : interaction.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from functools import partial

import torch
# ones_initializer = partial(constant_, val=1.0)
from torch import nn
from torch.nn.init import orthogonal_

from lightMolNet.Module.activations import shifted_softplus
from lightMolNet.Module.convolution import CFConv
from lightMolNet.Module.cutoff import CosineCutoff
from lightMolNet.Module.residual import ResidualLayer
from lightMolNet.Module.util import Dense, Aggregate

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


class AtomInteractionWithResidual(nn.Module):
    r"""simple interaction layers for modeling interactions of atomistic systems.

        Args
        ----
            n_atom_embeddings:int
                number of features to describe atomic environments.
            n_basis:int
                number of rbf or any other basis as expanded input.
            n_residual_atomic:int
                number of residual layers for atomics.
            num_residual_interaction:int
                number of residual layers for interaction.
            activation_fn:nn.Module
                activation function for calculating proceeds.

        References
        ----------
        .. [#physnet] Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting Energies,
                      Forces, Dipole Moments and Partial Charges" arxiv:1902.08408 (2019).
        """

    def __init__(self, n_atom_embeddings, n_basis, n_residual_atomic, num_residual_interaction, activation_fn=None):
        super(AtomInteractionWithResidual, self).__init__()
        self._n_atom_embeddings = n_atom_embeddings
        self._n_basis = n_basis
        self.activation_fn = activation_fn

        self.k2f = Dense(self.n_basis, self.n_atom_embeddings, bias=False)
        self.dense_i = Dense(self.n_atom_embeddings, self.n_atom_embeddings, activation=activation_fn,
                             weight_init=orthogonal_)
        self.dense_j = Dense(self.n_atom_embeddings, self.n_atom_embeddings, activation=activation_fn,
                             weight_init=orthogonal_)
        self.interact_res = nn.Sequential(*
                                          [
                                              ResidualLayer(self.n_atom_embeddings,
                                                            self.n_atom_embeddings,
                                                            activation=activation_fn
                                                            )
                                              for _ in range(num_residual_interaction)
                                          ]
                                          )
        self.dense_interact = Dense(self.n_atom_embeddings, self.n_atom_embeddings, weight_init=orthogonal_)
        self.u_gate = nn.Parameter(torch.ones(n_atom_embeddings))
        self.atom_res = nn.Sequential(*
                                      [
                                          ResidualLayer(self.n_atom_embeddings,
                                                        self.n_atom_embeddings,
                                                        activation=activation_fn
                                                        )
                                          for _ in range(n_residual_atomic)
                                      ]
                                      )
        self.agg = Aggregate(
            axis=2
        )

    def forward(self, x, rbf, neighbor):

        # interaction part
        # pre-activation
        if self.activation_fn is not None:
            xa = self.activation_fn(x)
        else:
            xa = x
        # feature from radial basis functions
        g = self.k2f(rbf)
        # calculate contribution of neighbors and central atom
        xi = self.dense_i(xa)

        nbh_size = neighbor.size()
        nbh = neighbor.view(-1, nbh_size[1] * nbh_size[2], 1).expand(-1, -1, xa.size(2))
        xj = torch.gather(xa, 1, nbh).view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        xj = xj * g
        xj = self.agg(xj)
        # TODO:test
        m = xi + xj
        if len(self.interact_res) > 0:
            m = self.interact_res(m)
        if self.activation_fn is not None:
            m = self.activation_fn(m)
        x = self.u_gate * x + self.dense_interact(m)
        # interaction part end
        # atom res part
        if len(self.atom_res) > 0:
            x = self.atom_res(x)
        return x

    @property
    def n_basis(self):
        return self._n_basis

    @property
    def n_atom_embeddings(self):
        return self._n_atom_embeddings
