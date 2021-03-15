# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : physnet.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn.init import constant_, orthogonal_

from lightMolNet import InputPropertiesList, Properties
from lightMolNet.Module.activations import shifted_softplus
from lightMolNet.Module.cutoff import RBFCutoff
from lightMolNet.Module.interaction import AtomInteractionWithResidual
from lightMolNet.Module.neighbors import AtomDistances
from lightMolNet.Module.residual import ResidualLayer
from lightMolNet.Module.util import Dense

zeros_initializer = partial(constant_, val=0.0)


class PhysNet(nn.Module):
    r"""
    References
    ----------
    .. [#PhysNet] Oliver T. Unke and Markus Meuwly
        PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments, and Partial Charges
        Journal of Chemical Theory and Computation 2019 15 (6), 3678-3693
        DOI: 10.1021/acs.jctc.9b00181


    """

    def __init__(self,
                 n_atom_embeddings: int = 128,  # dimensionality of feature vector of each element atom
                 n_rbf_basis: int = 128,  # number of radial basis functions
                 n_blocks: int = 5,  # number of blocks
                 cutoff: float = 10.0,  # cutoff of interactions
                 n_residual_atomic: int = 25,  # number of residual layers for atomic refinements of feature vector
                 n_residual_interaction=2,  # number of residual layers for refinement of message vector
                 n_residual_output=1,  # number of residual layers for the output blocks
                 max_Z: int = 100,  # scale of embedding matrix, due to maximum number of elements.
                 activation_fn: torch.nn.Module = shifted_softplus,
                 cal_distance=True,  # calculate distance matrix every time
                 use_electrostatic=True,  # adds electrostatic contributions to atomic energy
                 use_dispersion=True,  # adds dispersion contributions to atomic energy
                 kehalf=7.199822675975274,  # half (else double counting) of the Coulomb constant (default is in units e=1, eV=1, A=1)
                 s6=None,  # s6 coefficient for d3 dispersion, by default is learned
                 s8=None,  # s8 coefficient for d3 dispersion, by default is learned
                 a1=None,  # a1 coefficient for d3 dispersion, by default is learned
                 a2=None,  # a2 coefficient for d3 dispersion, by default is learned
                 Eshift=0.0,  # initial value for output energy shift (makes convergence faster)
                 Escale=1.0,  # initial value for output energy scale (makes convergence faster)
                 Qshift=0.0,  # initial value for output charge shift
                 Qscale=1.0,  # initial value for output charge scale
                 atomref=None
                 ):
        super(PhysNet, self).__init__()
        self._n_atom_embeddings = n_atom_embeddings
        self._n_rbf_basis = n_rbf_basis
        self._n_blocks = n_blocks
        self._cutoff = cutoff
        self._n_residual_atomic = n_residual_atomic
        self._n_residual_interaction = n_residual_interaction
        self._n_residual_output = n_residual_output
        self._max_Z = max_Z
        self.activation_fn = activation_fn
        self._kehalf = kehalf

        self.register_buffer("Eshift", Eshift * torch.ones(1))
        self.register_buffer("Escale", Escale * torch.ones(1))
        self.register_buffer("Qshift", Qshift * torch.ones(1))
        self.register_buffer("Qscale", Qscale * torch.ones(1))

        if atomref is not None:
            self._atom_ref_energy = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref[Properties.energy_U0].astype(np.float32))
            )
            # TODO:More atom refs.
        else:
            self._atom_ref_energy = None

        self._cal_distance = cal_distance
        self._use_electrostatic = use_electrostatic
        self._use_dispersion = use_dispersion

        # define blocks and layers
        self.embeddings = nn.Embedding(max_Z, n_atom_embeddings, padding_idx=0)
        self.distance_layer = AtomDistances()
        self.rbf_layer = RBFCutoff(self.n_rbf_basis, self._cutoff)
        self.interaction_block = nn.ModuleList(
            [
                AtomInteractionWithResidual(
                    self.n_atom_embeddings,
                    self.n_rbf_basis,
                    self.n_residual_atomic,
                    self.n_residual_interaction,
                    activation_fn=self.activation_fn
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.output_block = nn.ModuleList(
            [
                PhysNetOutputBlock(
                    n_atom_embeddings=self.n_atom_embeddings,
                    n_residual_output=self._n_residual_output,
                    activation_fn=activation_fn
                )
                for _ in range(self.n_blocks)
            ]
        )

    def forward(self, inputs: list):
        """Compute atomic representations/embeddings.

            Parameters
            ----------
                inputs: list
                    atomic_numbers:torch.Tensor,

                    positions:torch.Tensor,

                    cell:torch.Tensor,

                    cell_offset:torch.Tensor,

                    neighbors:torch.Tensor,

                    neighbor_mask:torch.Tensor

            Returns
            -------
                x:torch.Tensor
                    atom-wise representation.
                list of torch.Tensor:
                    intermediate atom-wise representations, if
                    return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[InputPropertiesList.Z]
        positions = inputs[InputPropertiesList.R]
        # cell = inputs[InputPropertiesList.cell]
        # cell_offset = inputs[InputPropertiesList.cell_offset]
        neighbors = inputs[InputPropertiesList.neighbors]
        neighbor_mask = inputs[InputPropertiesList.neighbor_mask]
        total_charge = inputs[InputPropertiesList.totcharge]
        if total_charge is None:
            total_charge = 0

        Dij = self.distance_layer(positions,
                                  neighbors,
                                  neighbor_mask=neighbor_mask)

        rbf = self.rbf_layer(Dij)
        x = self.embeddings(atomic_numbers)
        Ea = 0
        Qa = 0
        for i in range(self.n_blocks):
            x = self.interaction_block[i](x, rbf, neighbors)
            out = self.output_block[i](x)
            Ea = Ea + out[:, :, 0]
            Qa = Qa + out[:, :, 1]
            # out2=out**2

        Ea = self.Escale * Ea + \
             self.Eshift + \
             0 * torch.zeros_like(positions.sum(-1))
        Qa = self.Qscale * Qa + self.Qshift

        # scaled_charges
        Na = torch.sum(torch.where(atomic_numbers > 0, torch.ones_like(atomic_numbers), torch.zeros_like(atomic_numbers)), dim=1)
        Qa = Qa + (total_charge - torch.sum(Qa, dim=-1, keepdim=True)) / Na[:, None]

        if self._use_electrostatic:
            Qi = Qa[:, :, None]
            n_batch = Qa.size()[0]
            idx_m = torch.arange(n_batch,
                                 device=Qa.device,
                                 dtype=torch.long)[:, None, None]
            Qj = Qa[idx_m, neighbors[:, :, :]]
            switch = self._switch(Dij)
            cswitch = 1 - switch
            Eele_ordinary = torch.where(neighbor_mask == 1, 1 / Dij, torch.zeros_like(Dij))
            Eele_shielded = 1 / torch.sqrt(Dij * Dij + 1)
            Eele = self.kehalf * torch.einsum("bij,bij->bi", Qi * Qj, (cswitch * Eele_shielded + switch * Eele_ordinary))
            # TODO:test and verify
            Ea = Ea + Eele
        if self._use_dispersion:
            raise NotImplementedError
        D = 0

        # atom ref from calculations
        if self._atom_ref_energy is not None:
            Ei = self._atom_ref_energy(atomic_numbers).squeeze(-1)
            Ea = Ea + Ei
        return torch.einsum("bi->b", Ea).view(-1, 1), D

    def _switch(self, Dij):
        D = Dij
        cut = self.cutoff / 2
        Dr = D / cut
        D3 = Dr * Dr * Dr
        D4 = D3 * Dr
        D5 = D4 * Dr
        return torch.where(Dij < cut, 6 * D5 - 15 * D4 + 10 * D3, torch.ones_like(Dij))

    @property
    def n_atom_embeddings(self):
        return self._n_atom_embeddings

    @property
    def n_rbf_basis(self):
        return self._n_rbf_basis

    @property
    def n_blocks(self):
        return self._n_blocks

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def n_residual_atomic(self):
        return self._n_residual_atomic

    @property
    def n_residual_interaction(self):
        return self._n_residual_interaction

    @property
    def n_residual_output(self):
        return self._n_residual_output

    @property
    def max_Z(self):
        return self._max_Z

    @property
    def distance_layer(self):
        # TODO:calculate or not wrapper.
        return self._distance_layer

    @property
    def kehalf(self):
        return self._kehalf


class PhysNetOutputBlock(nn.Module):
    def __init__(self, n_atom_embeddings, n_residual_output, activation_fn):
        super(PhysNetOutputBlock, self).__init__()
        self.n_atom_embeddings = n_atom_embeddings
        self.n_residual_output = n_residual_output
        self.activation_fn = activation_fn

        self._res_layer = nn.Sequential(*
                                        [
                                            ResidualLayer(
                                                self.n_atom_embeddings,
                                                self.n_atom_embeddings,
                                                activation=activation_fn,
                                                bias=True
                                            )
                                            for _ in range(self.n_residual_output)
                                        ]
                                        )
        self._dense = Dense(in_features=self.n_atom_embeddings,
                            out_features=2,
                            bias=False,
                            weight_init=orthogonal_
                            )

    def forward(self, x):
        x = self._res_layer(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return self._dense(x)
