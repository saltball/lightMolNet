# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : dataloader.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch

from lightMolNet import Properties, InputPropertiesList_y, InputPropertiesList


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

    batch_list = [None for i in range(len(InputPropertiesList.input_list))]
    properties_list = [None for i in range(len(InputPropertiesList_y.input_list))]
    for index, pn in enumerate(batch):
        if pn in InputPropertiesList_y.input_list:
            properties_list[InputPropertiesList_y.input_list.index(pn)] = batch[pn]
        elif pn in InputPropertiesList.input_list:
            batch_list[InputPropertiesList.input_list.index(pn)] = batch[pn]
    return batch_list, properties_list


def _collate_aseatoms_with_cuda(examples):
    batch_list, properties_list = _collate_aseatoms(examples)
    for idx, k in enumerate(batch_list):
        if k is not None:
            batch_list[idx] = k[:].to(device="cuda")
    return batch_list, properties_list

def _collate_fn_using_dif_with_file(examples, diff_file):
    """
    return input dict with modified by `diff_file`(in database order)
    """
    difflist = np.load(diff_file)
    batch_list, properties_list = _collate_aseatoms(examples)
    properties_list[InputPropertiesList_y.energy_U0] = torch.Tensor(difflist[batch_list[InputPropertiesList.idx]]).reshape(properties_list[InputPropertiesList_y.energy_U0].size())
    return batch_list, properties_list


def _collate_fn_using_dif(diff_file):
    """
    wrapper of `_collate_fn_using_dif_with_file` for torch dataset moudle.
    """
    func = partial(_collate_fn_using_dif_with_file, diff_file=diff_file)
    return func