# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : schnetlong.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #


import pytorch_lightning as pl
import torch
from lightMolNet import Properties
from lightMolNet.Struct.nn.schnet import SchNetLong
from lightMolNet.data.atomsref import get_refatoms, refat_xTB
from lightMolNet.data.dataloader import _collate_aseatoms
from lightMolNet.datasets.LitDataSet.xtbxyzdataset import XtbXyzDataSet
from lightMolNet.net import LitNet

Batch_Size = 64
USE_GPU = 1

atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)


def cli_main(ckpt_path=None, schnetold=False):
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_0_loss_MAE',
        filename='FullNet-{epoch:02d}-{val_loss:.4f}',
        save_top_k=2,
        save_last=True
    )
    statistics = False
    dataset = XtbXyzDataSet(dbpath="fullerxtbdata2070.db",
                            xyzfiledir=r"D:\CODE\#DATASETS\FullDB\xTB2070",
                            atomref=atomrefs,
                            batch_size=Batch_Size,
                            pin_memory=True,
                            proceed=True,
                            statistics=statistics,
                            collate_fn=_collate_aseatoms
                            )
    dataset.prepare_data()
    dataset.setup(split_file_name="fuller2070")
    scheduler = {"_scheduler": torch.optim.lr_scheduler.CyclicLR,
                 "base_lr": 1e-9,
                 "max_lr": 1e-4,
                 "step_size_up": 10,
                 "step_size_down": 50,
                 "cycle_momentum": False
                 }
    model = LitNet(representNet=[SchNetLong],
                   batch_size=Batch_Size,
                   learning_rate=1e-5,
                   datamodule=dataset,
                   scheduler=scheduler)
    model._freeze_output()

    if ckpt_path is not None:
        from collections import OrderedDict
        state_dict = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if "represent" in k:
                if "embedding" in k:
                    name = ".".join(["represent.0.embeddings", *k.split(".")[2:]])
                else:
                    name = ".".join(["represent.0", *k.split(".")[1:]])
                if k == "representation.embedding.weight":
                    v = v[:18, :]
            elif "outputU0" in k:
                if 'outputU0.atomref.weight' in k:
                    name = "output.0.atomref.weight"
                    v = torch.Tensor(atomrefs["energy_U0"])
                elif 'standardize.mean' in k:
                    name = "output.0.standardize.mean"
                    v = v * 0
                elif 'standardize.stddev' in k:
                    name = "output.0.standardize.stddev"
                    v = v / v
                elif 'out_net' in k:
                    name = ".".join(["output.0.out_net", *k.split(".")[2:]])

            new_state_dict[name] = v
        if schnetold:
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict["state_dict"], strict=False)

        # model.freeze()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=USE_GPU,
        auto_lr_find=True,
        benchmark=True,
        max_epochs=10000,
        # auto_scale_batch_size='binsearch'
    )

    ### train
    trainer.fit(model)
    # torch.save(model.fullernet, 'save.pt')

    ### scale_batch
    # trainer.tune(model)

    ### lr_finder
    # lr_finder = trainer.tuner.lr_find(model, min_lr=1e-12, max_lr=0.5e-5)
    # fig = lr_finder.plot(suggest=True, show=True)
    # print(lr_finder.suggestion())

    # result = trainer.test(model, verbose=True)
    # print(result)


if __name__ == '__main__':
    scheduler = {"_scheduler": torch.optim.lr_scheduler.CyclicLR,
                 "base_lr": 1e-9,
                 "max_lr": 1e-4,
                 "step_size_up": 10,
                 "step_size_down": 50,
                 "cycle_momentum": False
                 }
    ckpt_path = r"E:\#Projects\#Research\0105-xTBQM9-SchNet-baseline-2\output01051532\checkpoints\FullNet-epoch=1750-val_loss=0.0000.ckpt"
    cli_main(ckpt_path=ckpt_path, schnetold=False)
