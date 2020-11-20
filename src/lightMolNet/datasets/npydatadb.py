# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : npydatadb.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import logging

import numpy as np
from ase.data import atomic_numbers
from ase.db import connect
from ase.units import Hartree
from tqdm import tqdm

from lightMolNet.datasets import FileSystemAtomsData

logger = logging.getLogger(__name__)


class npydatadb(FileSystemAtomsData):
    r"""
    establish database from numpy .npy store calculation files.

    Parameters
    ----------
    dbpath:str

    db_and_npy:dict

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
            refdbpath,
            npypath,
            zmax=18,
            refatom=None,
            subset=None,
            load_only=None,
            units=None,
            collect_triples=False,
            **kwargs
    ):
        if units is None:
            units = npydatadb.units
        available_properties = [
            npydatadb.U0
        ]
        self.filecontextdir = {
            "refdbpath": refdbpath,
            "npypath": npypath
        }
        self.refatom = refatom
        self.zmax = zmax

        super(npydatadb, self).__init__(
            dbpath=dbpath,
            filecontextdir=self.filecontextdir,
            subset=subset,
            available_properties=available_properties,
            load_only=load_only,
            units=units,
            collect_triples=collect_triples,
            **kwargs
        )

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return npydatadb(
            dbpath=self.dbpath,
            refdbpath=self.filecontextdir["refdbpath"],
            npypath=self.filecontextdir["npypath"],
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
            npydatadb.U0
        ]
        atref = np.zeros((self.zmax, len(labels)))
        for z in self.refatom.keys():
            atref[atomic_numbers[z], 0] = float(self.refatom[z]["U0"])  # Only energy

        return atref, labels

    def _load_data(self):
        all_atoms = []
        all_properties = []
        labels = [
            npydatadb.U0
        ]
        refdb = connect(self.filecontextdir['refdbpath'])
        npydata = np.load(self.filecontextdir['npypath'])
        for i in tqdm(range(1, len(self.refdb) + 1)):
            atoms = refdb.get_atoms(i)
            en = npydata[i - 1]
            all_atoms.append(atoms)
            properties = {}
            pn = self.available_properties[0]
            properties[pn] = np.array([en * self.units[pn]])
            all_properties.append(properties)

        logger.info("Write atoms to db...")
        self.add_systems(all_atoms, all_properties)

        logger.info("Done.")
