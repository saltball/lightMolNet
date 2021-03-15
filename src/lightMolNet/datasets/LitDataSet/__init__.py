# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : __init__.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from multiprocessing import cpu_count

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lightMolNet.data.dataloader import _collate_aseatoms
from lightMolNet.data.partitioning import random_split_partial
from lightMolNet.datasets.statistics import get_statistics


class LitDataSet(pl.LightningDataModule):
    def __init__(
            self,
            dbpath="LitDataSet.db",
            atomref=None,
            batch_size=10,
            num_workers=cpu_count(),
            pin_memory=False,
            statistics=True,
            valshuffle=False,
            **kwargs
    ):
        # if atomref is None:
        #     raise ValueError("Please define one specific atoms reference use `refatom=`."
        #                      "You can check `lightMolNet.data.atomsref` for details.")
        self.batch_size = batch_size
        self.dbpath = dbpath
        self.atomref = atomref
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.statistics = statistics
        self.means = None
        self.stddevs = None
        self.dataset = None
        self.valshuffle = valshuffle
        super().__init__()

    def setup(self, stage=None, data_partial=None, split_file_name="split"):
        if data_partial is None:
            data_partial = [60, 20, 20]
        self.train, self.val, self.test = \
            random_split_partial(data=self.dataset,
                                 partial=data_partial,
                                 split_file=split_file_name)
        if self.statistics:
            tmp_dataloader = DataLoader(self.train, batch_size=self.batch_size, collate_fn=_collate_aseatoms, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
            means, stddevs = get_statistics(tmp_dataloader,
                                            "energy_U0", single_atom_ref=self.atomref
                                            )
            print("In DataSets, we got mean={},std={}".format(means
                                                              , stddevs))
            self.means = means
            self.stddevs = stddevs

    def prepare_data(self):
        raise NotImplementedError(f"Method `prepare_data()` must be implemented for instance of class `{self.__class__}`")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=_collate_aseatoms,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=_collate_aseatoms,
            shuffle=self.valshuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=_collate_aseatoms,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def _all_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=_collate_aseatoms,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
