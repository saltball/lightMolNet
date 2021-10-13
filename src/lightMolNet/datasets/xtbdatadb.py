# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : xtbdatadb.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #
import json
from tqdm import tqdm
import pathlib
from lightMolNet.datasets.xyzdatadb import XYZDataDB
from ase.atoms import Atoms
import numpy as np

from lightMolNet.logger import DebugLogger

logger = DebugLogger(__name__)
INGORED = False
# INGORED for flag not to log multiple times.
from lightMolNet.datasets.xyzdatadb import simple_read_xyz_xtb


Add_Batch = 10000


class XTBDataDB(XYZDataDB):
    def __init__(
            self,
            dbpath,
            xtbjsonfiledir,
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
        if "available_properties" not in kwargs:
            kwargs["available_properties"] = [
                XYZDataDB.U0
            ]
        self.available_properties = kwargs["available_properties"]
        self.xtbjsonfiledir = xtbjsonfiledir
        self.refatom = refatom
        self.zmax = zmax
        super(XYZDataDB, self).__init__(
            dbpath=dbpath,
            filecontextdir=xtbjsonfiledir,
            subset=subset,
            load_only=load_only,
            units=units,
            collect_triples=collect_triples,
            **kwargs
        )

    def _load_data(self):
        all_atoms = []
        all_properties = []
        labels = [
            XYZDataDB.U0
        ]
        all_count = 0
        tbar = tqdm(pathlib.Path(self.xtbjsonfiledir).rglob("*.log"))
        for f in tbar:
            tbar.set_description_str("Working on File {}".format(f))
            with open(f, "r") as xyzf:
                file = simple_read_xyz_xtb(xyzf)
                for at in file:
                    pn = self.available_properties[0]
                    properties = dict({pn: np.array([at.info["energy"] * self.units[pn]])})  # Only energy
                    at.info = {}
                    all_atoms.append(at)
                    all_properties.append(properties)
                    all_count += 1
            if len(all_properties) > Add_Batch:
                self.add_systems(all_atoms, all_properties)
                tbar.set_postfix(Note=f"{len(all_properties)} molecules were just writen to db...")
                all_atoms = []
                all_properties = []

        if len(all_properties) > 0:
            self.add_systems(all_atoms, all_properties)
            logger.info("Write the last atoms to db...")
        logger.info(f"Done. Add {all_count} molecules to db file.")

def recursion_xyz_file(rootpath):
    for item in os.listdir(rootpath):
        if os.path.isdir(os.path.join(rootpath, item)):
            for file in recursion_xyz_file(os.path.join(rootpath, item)):
                yield file
        elif os.path.isfile(os.path.join(rootpath, item)):
            if os.path.splitext(item)[-1].endswith(".xyz"):
                yield os.path.join(rootpath, item)
            else:
                if not INGORED:
                    logger.debug("File [{}] is not one '.xyz' file,"
                             "it has been ignored.".format(os.path.join(rootpath, item)))
                    INGORED = True