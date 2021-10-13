# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : use_lmn_models_by_ase.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch
from lightMolNet import Properties
from lightMolNet.Struct.nn.schnet import SchNetLong
from lightMolNet.data.atomsref import get_refatoms, refat_xTB
from lightMolNet.data.dataloader import _collate_aseatoms_with_cuda, _collate_aseatoms
from lightMolNet.datasets.LitDataSet.xtbxyzdataset import XtbXyzDataSet
from lightMolNet.net import LitNet
from tqdm import tqdm

Batch_Size = 128
USE_GPU = 1
statistics = False

atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)

ckpt_path = r"E:\#Projects\#Research\0623-xtbfuller-SchNet-longinter\output20210623\lightning_logs\version_0\checkpoints\FullNet-epoch=662-val_loss=0.0000.ckpt"
dbpath = r"D:\CODE\PycharmProjects\lightMolNet\examples\schnetwithlong\fullerxtbdata2070.db"
xyzdir = r"D:\ "
predpath = "pred.npy"
refpath = "ref.npy"


def batch_use_torch_model(ckpt_path=ckpt_path, dbpath=dbpath, predsaveto=predpath, refsaveto=refpath,xyzdir=xyzdir):
    state_dict = torch.load(ckpt_path)

    scheduler = {"_scheduler": torch.optim.lr_scheduler.CyclicLR,
                 "base_lr": 1e-9,
                 "max_lr": 1e-4,
                 "step_size_up": 10,
                 "step_size_down": 50,
                 "cycle_momentum": False
                 }

    dataset = XtbXyzDataSet(dbpath=dbpath,
                            xyzfiledir=xyzdir,
                            atomref=atomrefs,
                            batch_size=Batch_Size,
                            # pin_memory=True,
                            proceed=True,
                            statistics=statistics,
                            collate_fn= _collate_aseatoms
                            )
    dataset.prepare_data()
    atomsDataLoader = dataset._all_dataloader()
    model = LitNet(representNet=[SchNetLong],
                   batch_size=Batch_Size,
                   learning_rate=1e-5,
                   datamodule=dataset,
                   scheduler=scheduler)
    model.load_state_dict(state_dict["state_dict"])
    if USE_GPU:
        model.to(device="cuda")
    model.freeze()

    predlist = []
    refenlist = []
    for inputss, keys in tqdm(atomsDataLoader, desc="Working on result calculation."):
        for idx, k in enumerate(inputss):
            if k is not None:
                inputss[idx] = k[:].to(device="cuda")

        results = model(inputss)["energy_U0"].detach().cpu().numpy()
        refen = keys[0].detach().cpu().numpy()
        preden = results
        predlist.extend(preden.tolist())
        refenlist.extend(refen.tolist())

    np.save(predsaveto, predlist)
    np.save(refsaveto, refenlist)


if __name__ == '__main__':
    ckpt_path = r"E:\#Projects\#Research\0623-xtbfuller-SchNet-longinter\output20210623\lightning_logs\version_0\checkpoints\FullNet-epoch=662-val_loss=0.0000.ckpt"
    dbpath = r"D:\CODE\PycharmProjects\lightMolNet\examples\schnetwithlong\fullerxtbdata7080.db"
    predpath = "pred7080.npy"
    refpath = "ref7080.npy"
    xyzdir= r"D:\CODE\#DATASETS\FullDB\xTB7080"
    batch_use_torch_model(ckpt_path=ckpt_path,
                          dbpath=dbpath,
                          predsaveto=predpath,
                          refsaveto=refpath,
                          xyzdir=xyzdir)
