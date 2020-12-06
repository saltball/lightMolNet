# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : G16DataSet.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from multiprocessing import cpu_count

from lightMolNet.datasets.G16datadb import G16datadb
from lightMolNet.datasets.LitDataSet import LitDataSet


class G16DataSet(LitDataSet):
    def __init__(
            self,
            dbpath="fullerxtb.db",
            logfiledir=None,
            atomref=None,
            batch_size=10,
            num_workers=cpu_count(),
            pin_memory=False,
            statistics=True,
            valshuffle=True,
            proceed=True
    ):
        super().__init__(
            dbpath=dbpath,
            atomref=atomref,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            statistics=statistics,
            valshuffle=valshuffle,
        )
        self.logfiledir = logfiledir
        self.proceed = proceed

    def prepare_data(self, stage=None):
        self.dataset = G16datadb(dbpath=self.dbpath,
                                 logfiledir=self.logfiledir,
                                 atomref=self.atomref,
                                 proceed=self.proceed
                                 )
