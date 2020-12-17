# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : atoms2input.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch

from lightMolNet import Properties, InputPropertiesList, InputPropertiesList_y
from lightMolNet.environment import SimpleEnvironmentProvider


def convert_atoms(
        atoms,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        centering_function=None,
        output=None,
):
    """
        Helper function to convert ASE atoms object to net input format.

        Parameters
        ----------
            atoms:ase.Atoms
                Atoms object of molecule
            environment_provider:callable
                Neighbor list provider.
            collect_triples:bool, optional
                Set to True if angular features are needed.
            centering_function:callable or None
                Function for calculating center of molecule (center of mass/geometry/...).
                Center will be subtracted from positions.
            output:dict
                Destination for converted atoms, if not None

        Returns
        -------
            dict of torch.Tensor:
            Properties including neighbor lists and masks reformated into net input format.
    """
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    cell = np.array(atoms.cell.array, dtype=np.float32)  # get cell array

    inputs[Properties.Z] = torch.LongTensor(atoms.numbers.astype(np.int))
    positions = atoms.positions.astype(np.float32)
    if centering_function:
        positions -= centering_function(atoms)
    inputs[Properties.R] = torch.FloatTensor(positions)
    inputs[Properties.cell] = torch.FloatTensor(cell)

    # get atom environment
    nbh_idx, offsets = environment_provider.get_environment(atoms)

    # Get neighbors and neighbor mask
    inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))

    # Get cells
    inputs[Properties.cell] = torch.FloatTensor(cell)
    inputs[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))

    inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
    mask = inputs[Properties.neighbors] >= 0
    inputs[Properties.neighbor_mask] = mask.float()
    inputs[Properties.neighbors] = (
            inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
    )

    batch_list = [None for i in range(len(InputPropertiesList.input_list))]
    properties_list = [None for i in range(len(InputPropertiesList_y.input_list))]
    for index, pn in enumerate(inputs):
        if pn in InputPropertiesList_y.input_list:
            properties_list[InputPropertiesList_y.input_list.index(pn)] = inputs[pn]
        elif pn in InputPropertiesList.input_list:
            batch_list[InputPropertiesList.input_list.index(pn)] = inputs[pn]
    return batch_list, properties_list
