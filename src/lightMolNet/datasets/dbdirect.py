# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : G16datadb.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import logging

import numpy as np
from ase.data import atomic_numbers
from ase.units import Hartree
from lightMolNet.datasets import AtomsData

logger = logging.getLogger(__name__)


class DBData(AtomsData):
    r"""
    establish database from g16 calculation files.

    Parameters
    ----------
    dbpath:str

    zmax:int
        maximum number of element embedding.

    refatom:dict
        references of atoms,like {"C":{"U0": -0.500273,"U":...}}

    """

    U0 = "energy_U0"
    units = [
        Hartree
    ]

    def __init__(
            self,
            dbpath,
            # logfiledir,
            zmax=18,
            refatom=None,
            subset=None,
            load_only=None,
            units=None,
            collect_triples=False,
            **kwargs
    ):
        if units is None:
            units = DBData.units
        available_properties = [
            DBData.U0
        ]
        # self.logfiledir = logfiledir
        self.refatom = refatom
        self.zmax = zmax

        super(DBData, self).__init__(
            dbpath=dbpath,
            # filecontextdir=logfiledir,
            subset=subset,
            available_properties=None,
            load_only=available_properties,
            units=units,
            collect_triples=collect_triples,
            **kwargs
        )

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return DBData(
            dbpath=self.dbpath,
            # logfiledir=self.logfiledir,
            zmax=self.zmax,
            refatom=self.refatom,
            subset=subidx,
            load_only=self.load_only,
            units=self.units,
            collect_triples=self.collect_triples
        )

    def _proceed(self):
        logger.info("Proceeding start...")
        self._load_data()
        logger.info("Data Loaded.")
        atref, labels = self._load_atomrefs()
        self.set_metadata({"atomrefs": atref.tolist(), "atref_labels": labels})
        logger.info("Atom references: Done.")

    def _load_atomrefs(self):
        labels = [
            DBData.U0
        ]
        atref = np.zeros((self.zmax, len(labels)))
        for z in self.refatom.keys():
            atref[atomic_numbers[z], 0] = float(self.refatom[z]["U0"])  # Only energy

        return atref, labels
