# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : dataloader.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch

from lightMolNet import Properties


def _collate_aseatoms(examples):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )

    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }
    has_atom_mask = Properties.atom_mask in batch.keys()
    has_neighbor_mask = Properties.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Properties.neighbor_mask] = torch.zeros_like(
            batch[Properties.neighbors]
        ).float()
    if not has_atom_mask:
        batch[Properties.atom_mask] = torch.zeros_like(batch[Properties.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Properties.neighbor_pairs_j in properties:
        batch[Properties.neighbor_pairs_mask] = torch.zeros_like(
            batch[Properties.neighbor_pairs_j]
        ).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Properties.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Properties.neighbor_mask][s] = mask
            batch[Properties.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Properties.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Properties.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Properties.neighbor_pairs_j]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.neighbor_pairs_mask][s] = nbh_idx_j >= 0

    return batch
