# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : module_utils.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from base64 import b64decode

import numpy as np
from ase.db import connect
from tqdm import tqdm


def read_deprecated_database(db_path):
    """
    Read all atoms and properties from deprecated ase databases.

    Args
    ----
        db_path:str
            Path to deprecated database

    Returns
    -------
        atoms:list
            All atoms objects of the database.
        properties:list
            All property dictionaries of the database.

    References
    ----------
    .. [#SchNetPack] K.T. Schütt, P. Kessel, M. Gastegger, K.A. Nicoli,
        A. Tkatchenko, K.-R. Müller.
        SchNetPack: A Deep Learning Toolbox For Atomistic Systems.
        Journal of Chemical Theory and Computation 15 (1), pp. 448-455. 2018.

    """
    with connect(db_path) as conn:
        db_size = conn.count()
    atoms = []
    properties = []

    for idx in tqdm(range(1, db_size + 1), "Reading deprecated database"):
        with connect(db_path) as conn:
            row = conn.get(idx)

        at = row.toatoms()
        pnames = [pname for pname in row.data.keys() if not pname.startswith("_")]
        props = {}
        for pname in pnames:
            try:
                shape = row.data["_shape_" + pname]
                dtype = row.data["_dtype_" + pname]
                prop = np.frombuffer(b64decode(row.data[pname]), dtype=dtype)
                prop = prop.reshape(shape)
            except Exception:
                # fallback for properties stored directly
                # in the row
                if pname in row:
                    prop = row[pname]
                else:
                    prop = row.data[pname]

                try:
                    prop.shape
                except AttributeError as e:
                    prop = np.array([prop], dtype=np.float32)
            props[pname] = prop

        atoms.append(at)
        properties.append(props)

    return atoms, properties
