# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : xtbgraddataset.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : xtbxyzdataset.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
from multiprocessing import cpu_count

from lightMolNet.datasets.LitDataSet import LitDataSet
from lightMolNet.datasets.xtbdatadb import XTBDataDB



class XtbGradDataSet(LitDataSet):
    def __init__(
            self,
            dbpath=None,
            xyzfiledir=None,  #
            atomref=None,
            batch_size=10,
            num_workers=cpu_count(),
            pin_memory=False,
            statistics=True,
            valshuffle=False,
            proceed=False,
            **kwargs
    ):
        super().__init__(
            dbpath=dbpath,
            atomref=atomref,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            statistics=statistics,
            valshuffle=valshuffle,
            **kwargs
        )
        self.xyzfiledir = xyzfiledir
        self.proceed = proceed

    def prepare_data(self, stage=None, **kwargs):
        self.dataset = XTBDataDB(dbpath=self.dbpath,
                                 xtbjsonfiledir=self.xyzfiledir,
                                 refatom=self.atomref,
                                 proceed=self.proceed,
                                 **kwargs
                                 )
