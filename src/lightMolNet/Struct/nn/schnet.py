# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : schnet.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import torch
from torch import nn

from lightMolNet import InputPropertiesList, InputPropertiesList_y
from lightMolNet.Module.atomcentersymfunc import GaussianSmearing
from lightMolNet.Module.cutoff import CosineCutoff
from lightMolNet.Module.interaction import SimpleAtomInteraction
from lightMolNet.Module.neighbors import AtomDistances


class SchNet(nn.Module):
    r"""
    References
    ----------
    .. [#SchNetPack] K.T. Schütt, P. Kessel, M. Gastegger, K.A. Nicoli,
            A. Tkatchenko, K.-R. Müller.
            SchNetPack: A Deep Learning Toolbox For Atomistic Systems.
            Journal of Chemical Theory and Computation 15 (1), pp. 448-455. 2018.
    """

    def __init__(self,
                 n_atom_embeddings: int = 128,
                 n_filters: int = 128,
                 n_interactions: int = 6,
                 cutoff: float = 10.0,
                 n_gaussians: int = 50,
                 normalize_filter: bool = False,
                 coupled_interactions: bool = False,
                 return_intermediate: bool = False,
                 max_Z: int = 18,
                 cutoff_network: torch.nn.Module = CosineCutoff,
                 trainable_gaussians: bool = False,
                 distance_expansion: torch.nn.Module = None,
                 # charged_systems: bool = False,
                 cal_distance=True
                 ):
        super(SchNet, self).__init__()
        # Embeddings for element, each of which is a vector of size (n_atom_embeddings)
        self.n_atom_embeddings = n_atom_embeddings
        self.embeddings = nn.Embedding(max_Z, n_atom_embeddings, padding_idx=0)

        # layer for computing inter-atomic distances
        if cal_distance:
            self.cal_distance = cal_distance
            self.distances = AtomDistances()

        # layer for expanding inter-atomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion = distance_expansion

        # layers for computing interaction
        if coupled_interactions:
            # use the same SimpleAtomInteraction instance for each kind of interaction
            self.interactions = nn.ModuleList(
                [
                    SimpleAtomInteraction(
                        n_atom_embeddings=n_atom_embeddings,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SimpleAtomInteraction instance for each kind of interaction
            self.interactions = nn.ModuleList(
                [
                    SimpleAtomInteraction(
                        n_atom_embeddings=n_atom_embeddings,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermediate = return_intermediate
        # self.charged_systems = charged_systems
        # if charged_systems:
        #     self.charge = nn.Parameter(torch.Tensor(1, n_atom_embeddings))
        #     self.charge.data.normal_(0, 1.0 / n_atom_embeddings ** 0.5)

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

        # get atom embeddings for the input atomic numbers
        x = self.embeddings(atomic_numbers)

        # charged system is not supported.

        # compute interatomic distance of every atom to its neighbors
        if getattr(self,"cal_distance",None):
            r_ij = self.distances(
                positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
            )
        else:
            r_ij = inputs[InputPropertiesList.distance]
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v
            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x


class SchNetLong(SchNet):

    def __init__(self,
                 max_Z=18,
                 n_atom_embeddings=128,
                 n_gaussians=12,
                 n_filters=5,
                 n_interactions=5,
                 cutoff=10,
                 n_long_interactions=5,
                 **kwargs):
        self.outputPro = [InputPropertiesList_y.energy_U0]
        super(SchNetLong, self).__init__(n_atom_embeddings=n_atom_embeddings,
                                         n_filters=n_filters,
                                         n_interactions=n_interactions,
                                         cutoff=cutoff,
                                         n_gaussians=n_gaussians,
                                         max_Z=max_Z,
                                         **kwargs)
        self._freeze_schnet()
        LongAtomInteraction = nn.ModuleList(
            [
                SimpleAtomInteraction(
                    n_atom_embeddings=n_atom_embeddings,
                    n_spatial_basis=n_gaussians,
                    n_filters=n_filters,
                    cutoff_network=CosineCutoff,
                    cutoff=cutoff,
                )
                for _ in range(n_long_interactions)
            ]
        )
        self.interactions_long = LongAtomInteraction

    def forward(self, inputs):
        atomic_numbers = inputs[InputPropertiesList.Z]
        positions = inputs[InputPropertiesList.R]
        cell = inputs[InputPropertiesList.cell]
        cell_offset = inputs[InputPropertiesList.cell_offset]
        neighbors = inputs[InputPropertiesList.neighbors]
        neighbor_mask = inputs[InputPropertiesList.neighbor_mask]

        # get atom embeddings for the input atomic numbers
        x = self.embeddings(atomic_numbers)

        # charged system is not supported.

        # compute interatomic distance of every atom to its neighbors
        if getattr(self,"cal_distance",None):
            r_ij = self.distances(
                positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
            )
        else:
            r_ij = inputs[InputPropertiesList.distance]
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        # for interaction in self.interactions:
        #     v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
        #     x = x + v
        #     if self.return_intermediate:
        #         xs.append(x)

        for interaction in self.interactions_long:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v
            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x

        # return self.fullernet(inputs)

    def _freeze_schnet(self):
        for param in self.parameters():
            param.requires_grad = False