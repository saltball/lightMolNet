# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : fullernet.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import pytorch_lightning as pl
import torch
from torch import nn

from lightMolNet import Properties, InputPropertiesList, InputPropertiesList_y
from lightMolNet.Module.GatherNet import MLP
from lightMolNet.Module.atomcentersymfunc import GaussianSmearing
from lightMolNet.Module.cutoff import CosineCutoff
from lightMolNet.Module.interaction import SimpleAtomInteraction
from lightMolNet.Module.neighbors import distance_matrix, AtomDistances
from lightMolNet.data.atomsref import get_refatoms, refat_xTB
from lightMolNet.data.dataloader import _collate_aseatoms
from lightMolNet.datasets.LitDataSet.xtbxyzdataset import XtbXyzDataSet
from lightMolNet.net import LitNetParent


class FullerNet(nn.Module):
    def __init__(self,
                 dis_cut=5,
                 max_Z=18,
                 n_atom_embeddings=128,
                 n_gaussians=6,
                 n_filters: int = 128,
                 n_interactions: int = 3,
                 cutoff: float = 5.0,
                 cutoff_network: torch.nn.Module = CosineCutoff,
                 normalize_filter: bool = False,
                 distance_expansion=None,
                 trainable_gaussians: bool = False,
                 ):
        super(FullerNet, self).__init__()
        # self.distance_layer = AtomDistances(return_unit_vec=True)
        self.embeddings = nn.Embedding(max_Z, n_atom_embeddings, padding_idx=0)
        self.distance_cutoff = CosineCutoff(cutoff=dis_cut)
        self.distances = AtomDistances()
        self.orbital_a_s = nn.ModuleList(
            [
                SimpleAtomInteraction(
                    n_atom_embeddings=n_atom_embeddings,
                    n_spatial_basis=n_gaussians,
                    n_filters=n_filters,
                    cutoff_network=cutoff_network,
                    cutoff=cutoff,
                    normalize_filter=normalize_filter,
                )
                for _ in range(n_interactions)
            ]
        )
        self.orbital_a_p = nn.ModuleList(
            [
                SimpleAtomInteraction(
                    n_atom_embeddings=n_atom_embeddings,
                    n_spatial_basis=n_gaussians,
                    n_filters=n_filters,
                    cutoff_network=cutoff_network,
                    cutoff=cutoff,
                    normalize_filter=normalize_filter,
                )
                for _ in range(n_interactions)
            ]
        )
        self.gather_a = MLP(n_atom_embeddings, 1)
        self.W = nn.Linear(1, 1, bias=True)
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )

    def forward(self, inputs: list):
        atomic_numbers = inputs[InputPropertiesList.Z]
        n_batch = atomic_numbers.size()[0]
        length = atomic_numbers.size()[1]
        positions = inputs[InputPropertiesList.R]
        cell = inputs[InputPropertiesList.cell]
        cell_offset = inputs[InputPropertiesList.cell_offset]
        neighbors = inputs[InputPropertiesList.neighbors]
        neighbor_mask = inputs[InputPropertiesList.neighbor_mask]
        total_charge = inputs[InputPropertiesList.totcharge]
        if total_charge is None:
            total_charge = 0

        # Geometry vectors

        Dij, Dvec = distance_matrix(positions,
                                    return_vecs=True)
        Dij_cut = self.distance_cutoff(Dij)  # distance cutoffed.
        orbital_p_vec = (Dij_cut[:, :, :, None] * Dvec).sum(-2)
        orbital_p_vec /= -torch.norm(orbital_p_vec, 2, 2)[:, :, None]  # p orbital vec normed.
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        x = self.embeddings(atomic_numbers)
        a_s = x
        a_p = x
        for orbital_a_s in self.orbital_a_s:
            v = orbital_a_s(a_s, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            a_s = a_s + v
        for orbital_a_p in self.orbital_a_p:
            v = orbital_a_p(a_p, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            a_p = a_p + v
        a_s = self.gather_a(a_s)
        a_p = self.gather_a(a_p)
        # a_ss = a_s[:, :, None] * a_s[:, None, :]
        a_pp = (a_p[:, :, None] * a_p[:, None, :]).squeeze(-1)
        a_sp = (a_s[:, :, None] * a_p[:, None, :]).squeeze(-1)
        spintor = -2 * a_sp ** 2 * Dij + 3 * a_sp
        pxpyintor = spintor + 4 * a_pp
        pzpzintor = pxpyintor + 1 / (Dij ** 2 - 1 / (2 * a_pp))
        orbital_p_cos = (orbital_p_vec[:, None, :] * orbital_p_vec[:, :, None]).sum(-1) / 2 + 0.5
        orbital_p_sin = 1 - orbital_p_cos

        beta = spintor * orbital_p_sin + 3 * orbital_p_cos * pzpzintor + 2 * orbital_p_sin * pxpyintor
        nelec = (atomic_numbers > 0).sum(-1, keepdim=True) // 2
        elecinx = torch.ones([n_batch, length], device=positions.device, dtype=int) * torch.arange(length,
                                                                                                   device=positions.device,
                                                                                                   dtype=torch.long).T
        elecinx_mask = elecinx < nelec

        E_pi = torch.symeig(beta, eigenvectors=True)[0]
        E_pi = (E_pi * elecinx_mask).sum(-1, keepdim=True)
        return self.W(E_pi)


class LitFullerNet(LitNetParent):
    def init(self, dis_cut=5,
             max_Z=18,
             n_atom_embeddings=128,
             n_gaussians=16,
             n_filters=6,
             n_interactions=5,
             cutoff=5, **kwargs):
        self.fullernet = FullerNet(dis_cut=dis_cut,
                                   max_Z=max_Z,
                                   n_atom_embeddings=n_atom_embeddings,
                                   n_gaussians=n_gaussians,
                                   n_filters=n_filters,
                                   n_interactions=n_interactions,
                                   cutoff=cutoff)
        self.outputPro = [Properties.energy_U0]

    def forward(self, inputs):
        outs = {}
        outs.update({Properties.energy_U0: self.fullernet(inputs)})
        return outs

        # return self.fullernet(inputs)


Batch_Size = 64
USE_GPU = 1

atomrefs = get_refatoms(refat_xTB, Properties.energy_U0, z_max=18)

import numpy as np
from functools import partial


def collate_fn_using_dif_with_file(examples, diff_file):
    difflist = np.load(diff_file)
    batch_list, properties_list = _collate_aseatoms(examples)
    properties_list[InputPropertiesList_y.energy_U0] = torch.Tensor(difflist[batch_list[InputPropertiesList.idx]]).reshape(properties_list[InputPropertiesList_y.energy_U0].size())
    return batch_list, properties_list


def collate_fn_using_dif(diff_file):
    func = partial(collate_fn_using_dif_with_file, diff_file=diff_file)
    return func


def cli_main(ckpt_path=None, schnetold=False):
    from pytorch_lightning.callbacks import ModelCheckpoint
    diff_file = r"D:\CODE\PycharmProjects\lightMolNet\examples\partialdata_20210425\FullXTB_dif.npy"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_0_loss_MAE',
        filename='FullNet-{epoch:02d}-{val_loss:.4f}',
        save_top_k=2,
        save_last=True
    )
    statistics = False
    dataset = XtbXyzDataSet(dbpath="fullerxtbdata60.db",
                            xyzfiledir=r"D:\CODE\#DATASETS\FullDB\xTBcal\C60",
                            atomref=atomrefs,
                            batch_size=Batch_Size,
                            pin_memory=True,
                            proceed=True,
                            statistics=statistics,
                            collate_fn=collate_fn_using_dif(diff_file)
                            )
    dataset.prepare_data()
    dataset.setup(data_partial=None)
    scheduler = {"_scheduler": torch.optim.lr_scheduler.CyclicLR,
                 "base_lr": 1e-7,
                 "max_lr": 1e-4,
                 "step_size_up": 10,
                 "step_size_down": 100,
                 "cycle_momentum": False
                 }
    model = LitFullerNet(learning_rate=1e-2,
                         datamodule=dataset,
                         batch_size=Batch_Size,
                         scheduler=scheduler,
                         # means=0,
                         # stddevs=1
                         )
    if ckpt_path is not None:
        from collections import OrderedDict
        state_dict = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if "represent" in k:
                if "embedding" in k:
                    name = ".".join(["represent.0.embeddings", *k.split(".")[2:]])
                else:
                    name = ".".join(["represent.0", *k.split(".")[1:]])
                if k == "representation.embedding.weight":
                    v = v[:18, :]
            elif "outputU0" in k:
                if 'outputU0.atomref.weight' in k:
                    name = "output.0.atomref.weight"
                    v = torch.Tensor(atomrefs["energy_U0"])
                elif 'standardize.mean' in k:
                    name = "output.0.standardize.mean"
                    v = v * 0
                elif 'standardize.stddev' in k:
                    name = "output.0.standardize.stddev"
                    v = v / v
                elif 'out_net' in k:
                    name = ".".join(["output.0.out_net", *k.split(".")[2:]])

            new_state_dict[name] = v
        if schnetold:
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict["state_dict"], strict=False)

        # model.freeze()
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=USE_GPU,
        auto_lr_find=True,
        benchmark=True,
        max_epochs=10000,
        # auto_scale_batch_size='binsearch'
    )

    ### train
    trainer.fit(model)
    # torch.save(model.fullernet, 'save.pt')

    ### scale_batch
    # trainer.tune(model)

    ### lr_finder
    # lr_finder = trainer.tuner.lr_find(model, min_lr=1e-12, max_lr=0.5e-5)
    # fig = lr_finder.plot(suggest=True, show=True)
    # print(lr_finder.suggestion())

    # result = trainer.test(model, verbose=True)
    # print(result)


if __name__ == '__main__':
    # ckpt_path = r"E:\#Projects\#Research\0105-xTBQM9-SchNet-baseline-2\output01051532\checkpoints\FullNet-epoch=1750-val_loss=0.0000.ckpt"
    cli_main(ckpt_path=None, schnetold=False)
    # from ase.atoms import Atoms
    # from ase.build import molecule
    # from ase.visualize import view
    # from lightMolNet.data.atoms2input import convert_atoms
    # from plotly.offline import plot
    # import plotly.graph_objs as go
    #
    # points = go.Scatter3d(x=[0, 5, 5, 10],
    #                       y=[5, 0, 10, 5],
    #                       z=[5, 0, 5, 0],
    #                       mode='markers',
    #                       marker=dict(size=2,
    #                                   color="rgb(227,26,28)")
    #                       )
    #
    # atoms = molecule("C60")
    # inputs, prop = convert_atoms(atoms)
    # for idx, k in enumerate(inputs):
    #     if k is not None:
    #         inputs[idx] = k[None, :]
    # model = FullerNet()
    # result = model(inputs)
    # print(result)
