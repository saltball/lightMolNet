# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : torchmdtest.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch
from ase.atoms import Atoms
from ase.optimize import LBFGS
from ase.units import Hartree
from ase.visualize import view

from lightMolNet import Properties
from lightMolNet.Struct.Atomistic.atomwise import Atomwise
from lightMolNet.Struct.nn import SchNet
from lightMolNet.caculator import torchCaculator
from lightMolNet.data.atomsref import get_refatoms
from lightMolNet.net import LitNet

ckpt_path = r"D:\CODE\PycharmProjects\lightMolNet\examples\load_model\last.ckpt"

state_dict = torch.load(ckpt_path)

Batch_Size = 32
USE_GPU = 1

refat_b3lypgd3 = {Properties.UNIT: {Properties.energy_U0: Hartree},
                  # "H": {Properties.energy_U0: -0.500273},
                  "C": {Properties.energy_U0: -37.843662},
                  # "N": {Properties.energy_U0: -54.583861},
                  # "O": {Properties.energy_U0: -75.064579},
                  # "F": {Properties.energy_U0: -99.718730}
                  }

atomrefs = get_refatoms(refat_b3lypgd3, Properties.energy_U0)
model = LitNet(learning_rate=1e-4,
               # datamodule=dataset,
               atomref=atomrefs,
               representNet=[SchNet],
               outputNet=[Atomwise],
               outputPro=[Properties.energy_U0],
               batch_size=Batch_Size,
               # scheduler=scheduler
               )

model.load_state_dict(state_dict["state_dict"])
model.freeze()
# model.to(device="cuda")
atoms = Atoms(numbers=np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
              positions=np.array([[1.123, 1.596, 0.195],
                                  [0.988, 0.928, 1.418],
                                  [1.463, -0.378, 1.252],
                                  [0.391, -1.257, 1.455],
                                  [0.157, -1.939, 0.255],
                                  [-1.123, -1.596, -0.195],
                                  [-0.988, -0.928, -1.418],
                                  [-1.463, 0.378, -1.252],
                                  [-0.391, 1.257, -1.455],
                                  [-0.157, 1.939, -0.255],
                                  [1.681, 0.703, -0.727],
                                  [0.745, 0.493, -1.747],
                                  [0.376, -0.858, -1.724],
                                  [1.084, -1.482, -0.69],
                                  [1.891, -0.518, -0.074],
                                  [-0.376, 0.858, 1.724],
                                  [-0.745, -0.493, 1.747],
                                  [-1.681, -0.703, 0.727],
                                  [-1.891, 0.518, 0.074],
                                  [-1.084, 1.482, 0.69]]),
              pbc=False)
view(atoms)
atoms.calc = torchCaculator(net=model)
# print(atoms.get_potential_energy())
# print(atoms.get_forces())
opt = LBFGS(atoms)
opt.run(fmax=0.001)
view(atoms)
