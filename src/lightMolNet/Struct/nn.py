# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : nn.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import torch
from torch import nn

from lightMolNet import Properties
from lightMolNet.Module.atomcentersymfunc import GaussianSmearing
from lightMolNet.Module.cutoff import CosineCutoff
from lightMolNet.Module.interaction import SimpleAtomInteraction
from lightMolNet.Module.neighbors import AtomDistances


class SchNet(nn.Module):
    r"""

    """

    def __init__(self,
                 n_atom_embeddings: int = 128,
                 n_filters: int = 128,
                 n_interactions: int = 3,
                 cutoff: float = 5.0,
                 n_gaussians: int = 25,
                 normalize_filter: bool = False,
                 coupled_interactions: bool = False,
                 return_intermediate: bool = False,
                 max_Z: int = 100,
                 cutoff_network: torch.nn.Module = CosineCutoff,
                 trainable_gaussians: bool = False,
                 distance_expansion: torch.nn.Module = None,
                 charged_systems: bool = False,
                 ):
        super(SchNet, self).__init__()
        # Embeddings for element, each of which is a vector of size (n_atom_embeddings)
        self.n_atom_embeddings = n_atom_embeddings
        self.embeddings = nn.Embedding(max_Z, n_atom_embeddings, padding_idx=0)

        # layer for computing inter-atomic distances
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
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_embeddings))
            self.charge.data.normal_(0, 1.0 / n_atom_embeddings ** 0.5)

    def forward(self, inputs):
        """Compute atomic representations/embeddings.

            Parameters
            ----------
                inputs:dict of torch.Tensor
                    dictionary of input tensors.

            Returns
            -------
                x:torch.Tensor
                    atom-wise representation.
                list of torch.Tensor:
                    intermediate atom-wise representations, if
                    return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # get atom embeddings for the input atomic numbers
        x = self.embeddings(atomic_numbers)

        if False and self.charged_systems and Properties.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Properties.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge

        # compute interatomic distance of every atom to its neighbors
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
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
