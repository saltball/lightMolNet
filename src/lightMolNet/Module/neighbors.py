# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : neighbors.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import torch
from torch import nn


def atom_distances(
        positions: torch.Tensor,
        neighbors: torch.Tensor,
        cell=None,
        cell_offsets=None,
        return_vecs: bool = False,
        normalize_vecs: bool = False,
        neighbor_mask: bool = None,
):
    r"""Compute distance of every atom to its neighbors.

    This function uses advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors.

    Args
    ----
        positions:torch.Tensor
            atomic Cartesian coordinates with #(N_at x 3) shape or
            (N_b x N_at x 3)
        neighbors:torch.Tensor
            indices of neighboring atoms with #(N_at x N_nbh) shape or
            (N_b x N_at x N_nbh)
        # cell:torch.Tensor, optional
            periodic cell of (N_b x 3 x 3) shape
        # cell_offsets:torch.Tensor, optional
            offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape
        return_vecs:bool, optional
            if True, also returns direction vectors.
        normalize_vecs: bool, optional
            if True, normalize direction vectors.
        neighbor_mask:torch.Tensor, optional
            boolean mask for neighbor positions.

    Returns
    -------
        (distances,dist_vec) : (torch.Tensor，torch.Tensor)
            distances: torch.Tensor
                distance of every atom to its neighbors with
                (N_b x N_at x N_nbh) shape.

            dist_vec: torch.Tensor
                direction cosines of every atom to its
                neighbors with (N_b x N_at x N_nbh x 3) shape (optional).

    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch,
                         device=positions.device,
                         dtype=torch.long)[:, None, None]
    # Get indices of neighboring atom positions
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]
    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset

    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    distances = torch.norm(dist_vec, 2, 3)

    if neighbor_mask is not None:
        # Avoid problems with zero distances in forces (instability of square
        # root derivative at 0) This way is neccessary, as gradients do not
        # work with inplace operations, such as e.g.
        # -> distances[mask==0] = 0.0
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances

    if return_vecs:
        tmp_distances = torch.ones_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]

        if normalize_vecs:
            dist_vec = dist_vec / tmp_distances[:, :, :, None]
        return distances, dist_vec
    return distances


class AtomDistances(nn.Module):
    r"""Layer for computing distance of every atom to its neighbors.

    Attributes
    ----------
        return_unit_vec (bool, optional): if True, the `forward` method also returns
            normalized direction vectors(unit vectors).

    """

    def __init__(
            self,
            return_unit_vec=False
    ):
        super(AtomDistances, self).__init__()
        self.return_unit_vec = return_unit_vec

    def forward(
            self,
            positions: torch.Tensor,
            neighbors: torch.Tensor,
            cell=None,
            cell_offsets=None,
            neighbor_mask: torch.Tensor = None
    ):
        """

        Parameters
        ----------
        positions:torch.Tensor
            atomic Cartesian coordinates with #(N_at x 3) shape or
            (N_b x N_at x 3)
        neighbors:torch.Tensor
            indices of neighboring atoms with #(N_at x N_nbh) shape or
           (N_b x N_at x N_nbh)
        # cell:torch.Tensor, optional
            periodic cell of (N_b x 3 x 3) shape
        # cell_offsets:torch.Tensor, optional
            offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape
        neighbor_mask:torch.Tensor, optional
            boolean mask for neighbor positions.

        Returns
        -------
        (distances,dist_vec) : (torch.Tensor，torch.Tensor)
            distances: torch.Tensor
                distance of every atom to its neighbors with
                (N_b x N_at x N_nbh) shape.

            dist_vec: torch.Tensor
                direction cosines of every atom to its
                neighbors with (N_b x N_at x N_nbh x 3) shape (optional).
        """

        return atom_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            return_vecs=self.return_unit_vec,
            normalize_vecs=True,
            neighbor_mask=neighbor_mask,
        )


def distance_matrix(positions, return_vecs=False):
    n_batch = positions.size()[0]
    atoms = positions.size()[1]
    idx_m = torch.arange(n_batch,
                         device=positions.device,
                         dtype=torch.long)[:, None, None]
    # Get indices of neighboring atom positions
    # pos_xyz = positions[idx_m, torch.stack([torch.stack([torch.arange(positions.size()[1],
    #                                                     device=positions.device,
    #                                                     dtype=torch.long) for _ in range(positions.size()[1])]
    #                                        )  for _ in range(n_batch)]), :]
    pos_xyz = positions[idx_m, torch.ones([n_batch, atoms, atoms], device=positions.device, dtype=int) * torch.arange(atoms,
                                                                                                                      device=positions.device,
                                                                                                                      dtype=torch.long), :]
    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]
    distances = torch.norm(dist_vec, 2, 3)
    if not return_vecs:
        return distances
    else:
        return distances, dist_vec / ((distances + 0.00001)[:, :, :, None])


if __name__ == '__main__':
    pos = torch.Tensor([
        [[0, 0, 0],
         [1, 1, 1],
         [-1, -1, 1],
         [-1, 1, -1]],
        [[0, 0, 0],
         [50, 50, 50],
         [-50, -50, -50],
         [-1, 1, -1]]
    ])
    print(distance_matrix(pos, return_vecs=True))
