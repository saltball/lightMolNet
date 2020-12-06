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
from ase.atoms import Atoms
from ase.data import atomic_numbers
from ase.units import Hartree
from tqdm import tqdm

from lightMolNet import Properties
from lightMolNet.datasets import FileSystemAtomsData

logger = logging.getLogger(__name__)


def simple_read_xyz_xtb(fileobj, index=None):
    lines = fileobj.readlines()
    natoms = int(lines[0])
    nimages = len(lines) // (natoms + 2)
    if index is None:
        index = slice(0, nimages)
    for i in range(*index.indices(nimages)):
        symbols = []
        positions = []
        n = i * (natoms + 2) + 2
        comments = {"energy": float(lines[n - 1].split()[1])}
        for line in lines[n:n + natoms]:
            symbol, x, y, z = line.split()[:4]
            symbol = symbol.lower().capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        yield Atoms(symbols=symbols, positions=positions, info=comments)


class XYZDataDB(FileSystemAtomsData):
    r"""
    establish database from g16 calculation files.

    Parameters
    ----------
    dbpath:str

    xyzfiledir:str

    zmax:int
        maximum number of element embedding.

    refatom:dict
        references of atoms,like {"C":{"U0": -0.500273,"U":...}}
        Unit: Hartree

    """

    U0 = Properties.energy_U0
    units = [
        Hartree
    ]

    def __init__(
            self,
            dbpath,
            xyzfiledir,
            zmax=18,
            refatom=None,
            subset=None,
            load_only=None,
            units=None,
            collect_triples=False,
            **kwargs
    ):
        if units is None:
            units = XYZDataDB.units
        available_properties = [
            XYZDataDB.U0
        ]
        self.xyzfiledir = xyzfiledir
        self.refatom = refatom
        self.zmax = zmax

        super(XYZDataDB, self).__init__(
            dbpath=dbpath,
            filecontextdir=xyzfiledir,
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

        return XYZDataDB(
            dbpath=self.dbpath,
            xyzfiledir=self.xyzfiledir,
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
            XYZDataDB.U0
        ]
        atref = np.zeros((self.zmax, len(labels)))
        for z in self.refatom.keys():
            atref[atomic_numbers[z], 0] = float(self.refatom[z]["U0"])  # Only energy

        return atref, labels

    def _load_data(self):
        all_atoms = []
        all_properties = []
        labels = [
            XYZDataDB.U0
        ]
        tbar = tqdm(recursion_xyz_file(self.xyzfiledir))
        for f in tbar:
            tbar.set_description_str("Working on File {}".format(f))
            properties = {}
            with open(f, "r") as xyzf:
                file = simple_read_xyz_xtb(xyzf)
                for at in file:
                    en = at.info["energy"]
                    at.info = {}
                    all_atoms.append(at)
                    pn = self.available_properties[0]
                    properties[pn] = np.array([en * self.units[pn]])
                    all_properties.append(properties)

        logger.info("Write atoms to db...")
        self.add_systems(all_atoms, all_properties)

        logger.info("Done.")


def recursion_xyz_file(rootpath):
    for item in os.listdir(rootpath):
        if os.path.isdir(os.path.join(rootpath, item)):
            for file in recursion_xyz_file(os.path.join(rootpath, item)):
                yield file
        elif os.path.isfile(os.path.join(rootpath, item)):
            if ".xyz" in os.path.splitext(item)[-1]:
                yield os.path.join(rootpath, item)
            else:
                logger.warning("File [{}] is not one '.xyz' file,"
                               "it has been ignored.".format(os.path.join(rootpath, item)))
