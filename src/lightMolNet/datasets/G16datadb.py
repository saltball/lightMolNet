# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : G16datadb.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import logging
import os

import numpy as np
from ase.units import Hartree
from lightMolNet.datasets import FileSystemAtomsData
from lightMolNet.datasets.fileprase import G16LogFiles

logger = logging.getLogger(__name__)


class G16datadb(FileSystemAtomsData):
    r"""
    establish database from g16 calculation files.

    Parameters
    ----------
    dbpath:str

    logfiledir:str

    zmax:int
        maximum number of element embedding.

    refatom:dict
        references of atoms,like {"U0":array,"U":array}

    """

    U0 = "energy_U0"
    units = [
        Hartree
    ]

    def __init__(
            self,
            dbpath,
            logfiledir,
            zmax=18,
            atomref=None,
            subset=None,
            load_only=None,
            units=None,
            collect_triples=False,
            **kwargs
    ):
        if units is None:
            units = G16datadb.units
        available_properties = [
            G16datadb.U0
        ]
        self.logfiledir = logfiledir
        self.atomref = atomref
        self.zmax = zmax

        super(G16datadb, self).__init__(
            dbpath=dbpath,
            filecontextdir=logfiledir,
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

        return G16datadb(
            dbpath=self.dbpath,
            logfiledir=self.logfiledir,
            zmax=self.zmax,
            atomref=self.atomref,
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
        labels = [str(i) for i in self.atomref.keys()]
        atref = []
        for i in self.atomref.keys():
            atref.append(self.atomref[i])
        return atref, labels

    def _load_data(self):
        all_atoms = []
        all_properties = []
        labels = [
            G16datadb.U0
        ]
        for item in os.listdir(self.logfiledir):
            if ".log" not in os.path.splitext(item)[-1] and ".out" not in os.path.splitext(item)[-1]:
                logger.warning(
                    "There is at least one file not '.log' or '.out' file,"
                    "Please make sure your data is clean."
                )
            elif ".log" or ".out" in os.path.splitext(item)[-1]:
                file = G16LogFiles(os.path.join(self.logfiledir, item))
                for at, en in zip(file.get_all_pairs()[0], file.get_all_pairs()[1]):
                    properties = {}
                    all_atoms.append(at)
                    pn = self.available_properties[0]
                    properties[pn] = np.array([en * self.units[pn]])
                    all_properties.append(properties)
        logger.info("Write atoms to db...")
        self.add_systems(all_atoms, all_properties)

        logger.info("Done.")
