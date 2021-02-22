# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : physnet.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
import torch
from torch import nn

from lightMolNet import InputPropertiesList
from lightMolNet.Module.activations import shifted_softplus


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
        self._activation_fn = activation_fn
        self._kehalf = kehalf

        self._cal_distance = cal_distance
        self._use_electrostatic = use_electrostatic
        self._use_dispersion = use_dispersion

        # define blocks and layers
        self._embeddings = nn.Embedding(max_Z, n_atom_embeddings, padding_idx=0)
        self._rbf_layer
        self._interaction_block
        self._output_block

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
        cell = inputs[InputPropertiesList.cell]
        cell_offset = inputs[InputPropertiesList.cell_offset]
        neighbors = inputs[InputPropertiesList.neighbors]
        neighbor_mask = inputs[InputPropertiesList.neighbor_mask]

        Ea, Qa, Dij_lr = self.atomic_properties(atomic_numbers,
                                                positions,
                                                )

        return x

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
    def activation_fn(self):
        return self._activation_fn

    @property
    def kehalf(self):
        return self._kehalf

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def rbf_layer(self):
        return self._rbf_layer

    @property
    def interaction_block(self):
        return self._interaction_block

    @property
    def output_block(self):
        return self._output_block
