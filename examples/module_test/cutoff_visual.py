# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : cutoff_visual.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from lightMolNet.Module.atomcentersymfunc import gaussian_smearing
from lightMolNet.Module.neighbors import atom_distances
from lightMolNet.data.atoms2input import convert_atoms
import torch
from ase.atoms import Atoms
from ase.build import molecule
import matplotlib.pyplot as plt

start = 0
stop = 5
n_gaussians = 50

# atoms = Atoms(positions=[[0, 0, 0],
#                          [1, 1, 1],
#                          [5, 5, 5]],
#               numbers=[6, 6, 6])
atoms = molecule("C60")

input = convert_atoms(atoms)
dis = atom_distances(input[0][1][None,], neighbors=input[0][4][None,])

offset1 = torch.linspace(0, 5, 50)
widths = torch.FloatTensor((offset1[1] - offset1[0]) * torch.ones_like(offset1))

# print(gaussian_smearing(dis, offset1, widths))
gs = gaussian_smearing(dis, offset1, widths)

print(dis)
for idx1, item1 in enumerate(dis):
    for idx2, it in enumerate(item1):
        for idx3, disi in enumerate(it):
            plt.scatter(offset1, gs[idx1, idx2, idx3])
plt.show()

offset2 = torch.linspace(5, 20, 150)
widths = torch.FloatTensor((offset2[1] - offset2[0]) * torch.ones_like(offset2))

# print(gaussian_smearing(dis, offset2, widths))
gs2 = gaussian_smearing(dis, offset2, widths)

print(dis)
for idx1, item1 in enumerate(dis):
    for idx2, it in enumerate(item1):
        for idx3, disi in enumerate(it):
            plt.scatter(offset2, gs2[idx1, idx2, idx3])
plt.show()
