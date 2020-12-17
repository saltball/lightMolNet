# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : torchmdcomparison.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import os

import matplotlib.pyplot as plt
import torch
from ase.optimize import GPMin
from ase.optimize.berny import Berny
from ase.units import Hartree, eV
from tqdm import tqdm

from lightMolNet import Properties
from lightMolNet.Struct.Atomistic.Atomwise import Atomwise
from lightMolNet.Struct.nn import SchNet
from lightMolNet.caculator import torchCaculator
from lightMolNet.data.atomsref import get_refatoms
from lightMolNet.datasets.fileprase import G16LogFiles
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

logdir = r"D:\CODE\PycharmProjects\lightMolNet\examples\logdata"

loglist = os.listdir(logdir)

reflist = []
predlist = []
difflist = []

tbar = tqdm(loglist)

plt.ion()
refenshow = None

for file in tbar:
    plt.clf()
    tbar.set_postfix(Current=f"{file}", RefEnergy=f"{refenshow}")
    f = G16LogFiles(os.path.join(logdir, file))
    atoms = f.get_ase_atoms([0])[0]
    atoms.pbc = False
    refat, refen = f.get_final_pairs()
    atoms.calc = torchCaculator(net=model)
    # print(atoms.get_potential_energy())
    # print(atoms.get_forces())
    opt = GPMin(atoms)
    opt.run(fmax=0.1)
    # opt = BFGSLineSearch(atoms)
    # opt.run(fmax=0.002)
    opt = Berny(atoms)
    opt.run(fmax=0.05)
    predat = atoms
    preden = atoms.get_potential_energy()
    reflist.append(refen[0] * eV / Hartree)
    refenshow = refen[0] * eV / Hartree
    predlist.append(preden)
    difflist.append(preden - refen[0] * eV / Hartree)
    plt.scatter(reflist, difflist, marker="x", s=10)
    plt.pause(0.4)

plt.ioff()
plt.show()
