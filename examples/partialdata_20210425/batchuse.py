# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : batchuse.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
import os

import numpy as np
import torch
from tqdm import tqdm

from lightMolNet import Properties
from lightMolNet.Struct.Atomistic.atomwise import Atomwise
from lightMolNet.Struct.nn.schnet import SchNet
from lightMolNet.data.atomsref import get_refatoms, refat_xTB
from lightMolNet.data.dataloader import _collate_aseatoms_with_cuda
from lightMolNet.datasets.LitDataSet.xtbxyzdataset import XtbXyzDataSet
from lightMolNet.net import LitNet
from examples.fullernet.fullernet import LitFullerNet

Batch_Size = 128
USE_GPU = 1


def calculate_qm9xtb(state_dict):
    atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)
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

    dataset = XtbXyzDataSet(dbpath=r"D:\CODE\#DATASETS\myDataSet\qm9inxtb.db",
                            xyzfiledir=r"D:\CODE\#DATASETS\FullDB\xTBBack",
                            # atomref=atomrefs,
                            # batch_size=Batch_Size,
                            # pin_memory=True,
                            # proceed=True,
                            )

    atomsDataLoader = dataset._all_dataloader()
    predlist = []
    refenlist = []
    for inputss, keys in tqdm(atomsDataLoader, desc="Working on result calculation."):
        results = model(inputss)["energy_U0"].detach().cpu().numpy()
        refen = keys[0].detach().cpu().numpy()
        preden = results
        predlist.extend(preden.tolist())
        refenlist.extend(refen.tolist())

    np.save("QM9XTB_pred.npy", predlist)
    np.save("QM9XTB_ref.npy", refenlist)


def calculate_fullerxtb(state_dict):
    atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)
    model = LitNet(learning_rate=1e-4,
                   datamodule=None,
                   atomref=atomrefs,
                   representNet=[SchNet],
                   outputNet=[Atomwise],
                   outputPro=[Properties.energy_U0],
                   batch_size=Batch_Size,
                   scheduler=None
                   )

    model.load_state_dict(state_dict["state_dict"])
    model.freeze()
    if USE_GPU:
        model.to(device="cuda")

    dataset = XtbXyzDataSet(dbpath=r"D:\CODE\#DATASETS\FullDB\subcal\C6070xtb.db",
                            xyzfiledir=r"D:\CODE\#DATASETS\FullDB\subcal",
                            atomref=atomrefs,
                            statistics=False,
                            # pin_memory=True,
                            proceed=True,
                            batch_size=Batch_Size,
                            num_workers=0,
                            collate_fn=_collate_aseatoms_with_cuda
                            )
    dataset.prepare_data()
    dataset.setup()
    atomsDataLoader = dataset._all_dataloader()
    predlist = []
    refenlist = []
    for inputss, keys in tqdm(atomsDataLoader, desc="Working on result calculation."):
        results = model(inputss)["energy_U0"].detach().cpu().numpy()
        refen = keys[0].detach().cpu().numpy()
        preden = results
        predlist.extend(preden.tolist())
        refenlist.extend(refen.tolist())

    np.save("D:\CODE\#DATASETS\FullDB\subcal\FullXTB_pred.npy", predlist)
    np.save("D:\CODE\#DATASETS\FullDB\subcal\FullXTB_ref.npy", refenlist)


def calculate_fullernet(state_dict1):
    atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)
    dataset = XtbXyzDataSet(dbpath=r"D:\CODE\#DATASETS\FullDB\subcal\C6070xtb.db",
                            xyzfiledir=r"D:\CODE\#DATASETS\FullDB\subcal",
                            atomref=atomrefs,
                            statistics=False,
                            # pin_memory=True,
                            proceed=True,
                            batch_size=Batch_Size,
                            num_workers=0,
                            collate_fn=_collate_aseatoms_with_cuda
                            )
    dataset.prepare_data()
    dataset.setup()
    atomsDataLoader = dataset._all_dataloader()
    model = LitFullerNet(learning_rate=1e-4,
                         datamodule=dataset,
                         atomref=atomrefs,
                         representNet=[SchNet],
                         outputNet=[Atomwise],
                         outputPro=[Properties.energy_U0],
                         batch_size=Batch_Size,
                         scheduler=None
                         )

    model.load_state_dict(state_dict1["state_dict"])
    model.freeze()
    if USE_GPU:
        model.to(device="cuda")

    predlist = []
    refenlist = []
    for inputss, keys in tqdm(atomsDataLoader, desc="Working on result calculation."):
        results = model(inputss)["energy_U0"].detach().cpu().numpy()
        refen = keys[0].detach().cpu().numpy()
        preden = results
        predlist.extend(preden.tolist())
        refenlist.extend(refen.tolist())

    np.save("D:\CODE\#DATASETS\FullDB\subcal\FullXTBgraph_pred.npy", predlist)
    # np.save("D:\CODE\#DATASETS\FullDB\subcal\FullXTBgraph_ref.npy", refenlist)


if __name__ == '__main__':
    ckpt_path = r"E:\#Projects\#Research\0105-xTBQM9-SchNet-baseline-2\output01051532\checkpoints\FullNet-epoch=1750-val_loss=0.0000.ckpt"
    ckpt_path1 = r"D:\CODE\PycharmProjects\lightMolNet\examples\fullernet\lightning_logs\version_1\checkpoints\FullNet-epoch=981-val_loss=0.0000.ckpt"
    state_dict = torch.load(ckpt_path)
    state_dict1 = torch.load(ckpt_path1)
    if not os.path.exists(r"QM9XTB_ref.npy"):
        calculate_qm9xtb(state_dict)
    # predlist=np.load("QM9XTB_pred.npy")
    # refenlist=np.load("QM9XTB_ref.npy")
    if not os.path.exists(r"D:\CODE\#DATASETS\FullDB\subcal\FullXTB_pred.npy"):
        calculate_fullerxtb(state_dict)
    predlist = np.load("D:\CODE\#DATASETS\FullDB\subcal\FullXTB_pred.npy")
    refenlist = np.load("D:\CODE\#DATASETS\FullDB\subcal\FullXTB_ref.npy")
    # difflist = np.save("D:\CODE\#DATASETS\FullDB\subcal\FullXTB_dif.npy", predlist - refenlist)

    if not os.path.exists(r"D:\CODE\#DATASETS\FullDB\subcal\FullXTBgraph_pred.npy"):
        calculate_fullernet(state_dict1)
    netpredlist = np.load("D:\CODE\#DATASETS\FullDB\subcal\FullXTBgraph_pred.npy")
    from matplotlib import pyplot as plt

    plt.scatter(predlist - netpredlist, refenlist, marker="x")
    plt.show()
