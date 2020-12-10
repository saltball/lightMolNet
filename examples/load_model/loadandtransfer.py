# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : loadandtransfer.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import time

import numpy as np
import torch
from ase.db import connect
from ase.units import Hartree
from lightMolNet import Properties
from lightMolNet.Struct.Atomistic.Atomwise import Atomwise
from lightMolNet.Struct.nn import SchNet
from lightMolNet.data.atomsref import get_refatoms
from lightMolNet.datasets.LitDataSet.G16DataSet import G16DataSet
from lightMolNet.environment import SimpleEnvironmentProvider
from lightMolNet.net import LitNet
from tqdm import tqdm

conn = connect(r"fullerxtb.db")


def get_atom(idx, conn=conn):
    return conn.get_atoms(idx)


def _convert_atoms(
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

    return inputs


def get_input(idx, conn):
    result = _convert_atoms(get_atom(idx, conn))
    result["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))
    for k, v in result.items():
        result[k] = v[None, :].to(device="cuda")
    return result


Batch_Size = 32
USE_GPU = 1

refat_b3lypgd3 = {Properties.UNIT: {Properties.energy_U0: Hartree},
                  # "H": {Properties.energy_U0: -0.500273},
                  "C": {Properties.energy_U0: -37.843662},
                  # "N": {Properties.energy_U0: -54.583861},
                  # "O": {Properties.energy_U0: -75.064579},
                  # "F": {Properties.energy_U0: -99.718730}
                  }

atomrefs = get_refatoms(refat_b3lypgd3, Properties.energy_U0)


def cli_main(ckpt_path):
    state_dict = torch.load(ckpt_path)
    model = LitNet(learning_rate=1e-4,
                   # datamodule=dataset,
                   atomref=atomrefs,
                   representNet=[SchNet],
                   outputNet=[Atomwise],
                   outputPro=[Properties.energy_U0],
                   batch_size=Batch_Size,
                   # scheduler=scheduler
                   )

    model.load_state_dict(state_dict["state_dict"])
    model.freeze()

    dataset = G16DataSet(dbpath="fullerxtb.db",
                         logfiledir=r"D:\CODE\PycharmProjects\lightMolNet\examples\logdata",
                         atomref=atomrefs,
                         batch_size=Batch_Size,
                         pin_memory=True)
    dataset.prepare_data()
    dataset.setup(data_partial=None)

    model.to(device="cuda")

    error = 0
    MRSE = 0
    delta = np.inf
    tbar = tqdm(range(1, len(conn) + 1))
    timecost = np.inf

    pred = np.array([])
    refs = np.array([])

    for idx in tbar:
        tbar.set_postfix(delta=delta, MAE=error, MRSE=MRSE, time=timecost)
        inputAtom = get_input(idx, conn)
        st = time.time()
        result = model(inputAtom)
        en = time.time()
        timecost = en - st
        result = result["energy_U0"].cpu()[0]
        pred = np.append(pred, result)
        refe = conn[idx].data["energy_U0"]
        refs = np.append(refs, refe)
        delta = np.abs(result - refe)
        error = (error * (idx - 1) + delta) / idx
        MRSE = np.sqrt((MRSE ** 2 * (idx - 1) + delta ** 2) / idx)

    np.savez(file="FullDFT", pred=pred, refs=refs)


if __name__ == '__main__':
    cli_main(ckpt_path="last.ckpt")
