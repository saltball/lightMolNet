# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : atomsref.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
from ase.data import atomic_numbers
from ase.units import Hartree, eV

from lightMolNet import Properties

refat_xTB = {Properties.UNIT: {Properties.energy_U0: eV},
             "H": {Properties.energy_U0: -10.707211383396714},
             "C": {Properties.energy_U0: -48.847445262804705},
             "N": {Properties.energy_U0: -71.00681805517411},
             "O": {Properties.energy_U0: -102.57117256025786},
             "F": {Properties.energy_U0: -125.69864294466228}
             }

refat_qm9 = {Properties.UNIT: {Properties.energy_U0: Hartree},
             "H": {Properties.energy_U0: -0.500273},
             "C": {Properties.energy_U0: -37.846772},
             "N": {Properties.energy_U0: -54.583861},
             "O": {Properties.energy_U0: -75.064579},
             "F": {Properties.energy_U0: -99.718730}
             }


def get_refatoms(refat: dict, properties: str or list or tuple = None, z_max=118):
    r"""

    Parameters
    ----------
    refat:dict
        like
        refat_qm9 = {Properties.UNIT: {Properties.energy_U0: Hartree,...},
             "H": {Properties.energy_U0: -0.500273,...},
             "C": {Properties.energy_U0: -37.846772},
             "N": {Properties.energy_U0: -54.583861},
             "O": {Properties.energy_U0: -75.064579},
             "F": {Properties.energy_U0: -99.718730},
             ...
             }
    properties:str or list or tuple
        default to litmolnet.Properties.energy_U0
    z_max:int
        maximum of atomic number.
        default to 118
    Returns
    -------
        dict:
            {Properties.energy_U0:array(118,1)}
    """
    if properties is None:
        properties = (Properties.energy_U0,)
    elif type(properties) is str:
        properties = (properties,)
    else:
        raise TypeError("Type {} is not supported for refatom properties input.".format(type(properties)))
    atomrefarray = {}
    for pn in properties:
        if pn not in refat[Properties.UNIT].keys():
            raise KeyError("Key {} not in atomref {}. Check your code.".format(pn, refat.__repr__()))
        else:
            atomrefarray.setdefault(pn, np.zeros([z_max, 1]))
    for atomsymbol, v in refat.items():
        if atomsymbol != Properties.UNIT:
            for pn in v.keys():
                if pn in atomrefarray.keys():
                    atomrefarray[pn][atomic_numbers[atomsymbol]] = refat[atomsymbol][pn] * refat[Properties.UNIT][pn] / Properties.properties_unit[pn]
    return atomrefarray
