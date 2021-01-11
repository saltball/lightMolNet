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
from ase.units import Hartree, eV
from tqdm import tqdm

from lightMolNet import Properties, InputPropertiesList
from lightMolNet.Struct.Atomistic.atomwise import Atomwise
from lightMolNet.Struct.nn import SchNet
from lightMolNet.data.atoms2input import convert_atoms
from lightMolNet.data.atomsref import get_refatoms, refat_xTB
from lightMolNet.datasets.LitDataSet.G16DataSet import G16DataSet
from lightMolNet.net import LitNet

conn = connect(r"fullerxtb.db")


def get_atom(idx, conn=conn):
    return conn.get_atoms(idx)

def get_input(idx, conn):
    result, ref = convert_atoms(get_atom(idx, conn))
    result[InputPropertiesList.idx] = torch.LongTensor(np.array([idx], dtype=np.int))
    for idx, k in enumerate(result):
        if k is not None:
            result[idx] = k[None, :].to(device="cuda")
    for idx, k in enumerate(ref):
        if k is not None:
            ref[idx] = k[None, :].to(device="cuda")
    return result, ref


Batch_Size = 32
USE_GPU = 1

refat_b3lypgd3 = {Properties.UNIT: {Properties.energy_U0: Hartree},
                  # "H": {Properties.energy_U0: -0.500273},
                  "C": {Properties.energy_U0: -37.843662},
                  # "N": {Properties.energy_U0: -54.583861},
                  # "O": {Properties.energy_U0: -75.064579},
                  # "F": {Properties.energy_U0: -99.718730}
                  }

atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)

global model, input_sample


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
                         # logfiledir=r"D:\CODE\PycharmProjects\lightMolNet\examples\logdata",
                         atomref=atomrefs,
                         batch_size=Batch_Size,
                         statistics=False,
                         pin_memory=True,
                         proceed=False)
    dataset.prepare_data()
    dataset.setup(data_partial=None)

    model.to(device="cuda")

    error = 0
    MRSE = 0
    delta = np.inf
    tbar = tqdm(range(1, len(conn) + 1))
    timecost = np.inf
    #
    pred = np.array([])
    refs = np.array([])

    for idx in tbar:
        tbar.set_postfix(delta=delta, MAE=error, MRSE=MRSE, time=timecost)
        inputAtom, ref = get_input(idx, conn)
        NAtom = len(inputAtom[0][InputPropertiesList.Z])
        st = time.time()
        result = model(inputAtom)
        en = time.time()
        timecost = en - st
        result = result["energy_U0"].cpu()[0] - NAtom * refat_xTB["C"][Properties.energy_U0]
        pred = np.append(pred, result)
        refe = conn[idx].data["energy_U0"] - NAtom * refat_b3lypgd3["C"][Properties.energy_U0] * Hartree / eV
        refs = np.append(refs, refe)
        delta = np.abs(result - refe)
        error = (error * (idx - 1) + delta) / idx
        MRSE = np.sqrt((MRSE ** 2 * (idx - 1) + delta ** 2) / idx)

    np.savez(file="trandferfromxtb", pred=pred, refs=refs)

    # TO ONNX
    # input_sample, ref = get_input(1, conn)
    # model(input_sample)
    # print(input_sample[InputPropertiesList.cell_offset].shape)
    # print(get_atom(1).__dict__)

    # dynamic_axes = {'inputs': {0: 'batch_size'},
    #                 "output": {0: "batch_size"}
    #                 }
    #
    # torch.onnx.export(
    #     model,
    #     input_sample,
    #     "model.onnx",
    #     input_names=["inputs", "keys"],
    #     output_names=["output"],
    #     opset_version=11,
    #     do_constant_folding=True,
    #     dynamic_axes=dynamic_axes
    # )


if __name__ == '__main__':
    cli_main(ckpt_path=r"E:\#Projects\#Research\0109-xtbfuller-SchNet-baseline-1\output\checkpoints\FullNet-epoch=22-val_loss=0.0000.ckpt")
