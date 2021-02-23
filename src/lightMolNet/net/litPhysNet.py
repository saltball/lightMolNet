# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : litPhysNet.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import pytorch_lightning as pl
import torch

import lightMolNet.Module.functional as F
from lightMolNet.Struct.nn.physnet import PhysNet
from lightMolNet.logger import InfoLogger

logger = InfoLogger(__name__)


class LitPhysNet(pl.LightningModule):
    def __init__(
            self,
            learning_rate=1e-4,
            batch_size=32,
            datamodule=None,
            n_atom_embeddings: int = 128,  # dimensionality of feature vector of each element atom
            n_rbf_basis: int = 64,  # number of radial basis functions
            n_blocks: int = 5,  # number of blocks
            cutoff: float = 10.0,  # cutoff of interactions
            n_residual_atomic: int = 2,  # number of residual layers for atomic refinements of feature vector
            n_residual_interaction=3,  # number of residual layers for refinement of message vector
            n_residual_output=1,  # number of residual layers for the output blocks
            max_Z: int = 100,  # scale of embedding matrix, due to maximum number of elements.
            cal_distance=True,  # calculate distance matrix every time
            use_electrostatic=True,  # adds electrostatic contributions to atomic energy
            use_dispersion=False,  # adds dispersion contributions to atomic energy
            kehalf=7.199822675975274,  # half (else double counting) of the Coulomb constant (default is in units e=1, eV=1, A=1)
            s6=None,  # s6 coefficient for d3 dispersion, by default is learned
            s8=None,  # s8 coefficient for d3 dispersion, by default is learned
            a1=None,  # a1 coefficient for d3 dispersion, by default is learned
            a2=None,  # a2 coefficient for d3 dispersion, by default is learned
            Eshift=0.0,  # initial value for output energy shift (makes convergence faster)
            Escale=1.0,  # initial value for output energy scale (makes convergence faster)
            Qshift=0.0,  # initial value for output charge shift
            Qscale=1.0,  # initial value for output charge scale
            scheduler=None
    ):
        """

        Parameters
        ----------
        learning_rate
        datamodule
        representNet
        n_atom_embeddings
        n_filters
        n_interactions
        cutoff
        n_gaussians
        max_Z
        outputNet
        outputPro
        batch_size
        atomref
        means
        stddevs
        scheduler

        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        super().__init__()
        self.save_hyperparameters("batch_size", "learning_rate")
        self.datamodule = datamodule

        self.net = PhysNet(
            n_atom_embeddings=n_atom_embeddings,  # dimensionality of feature vector of each element atom
            n_rbf_basis=n_rbf_basis,  # number of radial basis functions
            n_blocks=n_blocks,  # number of blocks
            cutoff=cutoff,  # cutoff of interactions
            n_residual_atomic=n_residual_atomic,  # number of residual layers for atomic refinements of feature vector
            n_residual_interaction=n_residual_interaction,  # number of residual layers for refinement of message vector
            n_residual_output=n_residual_output,  # number of residual layers for the output blocks
            max_Z=max_Z,  # scale of embedding matrix, due to maximum number of elements.

            cal_distance=cal_distance,  # calculate distance matrix every time
            use_electrostatic=use_electrostatic,  # adds electrostatic contributions to atomic energy
            use_dispersion=use_dispersion,  # adds dispersion contributions to atomic energy
            kehalf=kehalf,  # half (else double counting) of the Coulomb constant (default is in units e=1, eV=1, A=1)
            s6=None,  # s6 coefficient for d3 dispersion, by default is learned
            s8=None,  # s8 coefficient for d3 dispersion, by default is learned
            a1=None,  # a1 coefficient for d3 dispersion, by default is learned
            a2=None,  # a2 coefficient for d3 dispersion, by default is learned
            Eshift=Eshift,  # initial value for output energy shift (makes convergence faster)
            Escale=Escale,  # initial value for output energy scale (makes convergence faster)
            Qshift=Qshift,  # initial value for output charge shift
            Qscale=Qscale,  # initial value for output charge scale
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if scheduler is not None:
            schedulerpara = {k: v for k, v in scheduler.items() if k != "_scheduler"}
            self.scheduler = scheduler["_scheduler"](
                self.optimizer,
                **schedulerpara
            )

    def forward(
            self,
            inputs
    ):
        """

        Parameters
        ----------
        inputs:list
            check order of `lightMolNet.InputPropertiesList`

        Returns
        -------

        """
        # representation parts
        outs = self.net(inputs)
        return outs

    def training_step(self, batch, batch_idx):
        batch, y = batch

        self.log("lr", self.optimizer.state_dict()['param_groups'][0]['lr'], on_epoch=True)

        outs = self.forward(batch)
        loss = F.mae_loss_for_train(outs[0][:, None], y[0])
        # Logging to TensorBoard by default
        self.log('train_{}_loss_MAE'.format(0), loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return torch.mean(loss)

    def validation_step(self, batch, batch_idx):
        batch, y = batch
        outs = self.forward(batch)
        outs = outs[0]
        loss = F.mae_loss_for_metric(outs, y[0])
        # Logging to TensorBoard by default
        self.log('val_{}_loss_MAE'.format(0), loss, on_step=True, logger=True)
        return torch.mean(loss).detach()

    def test_step(self, batch, batch_idx):
        batch, y = batch
        Nouts = len(self.outputPro)

        outs = self.forward(batch)
        loss = torch.zeros([Nouts], device=self.device)
        test_result = {}
        for outPro in range(Nouts):
            test_result.update({f"pred{outPro}": None,
                                f"ref{outPro}": None})
            loss[outPro] = F.mae_loss_for_metric(outs[self.outputPro[outPro]], y[outPro])
            # Logging to TensorBoard by default
            self.log('test_{}_loss_MAE'.format(outPro), loss[outPro], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            test_result.update({f"pred{outPro}": outs[self.outputPro[outPro]].cpu(),
                                f"ref{outPro}": y[outPro].cpu()})
        return test_result

    def test_epoch_end(self, outputs):
        # np.save("outputs",outputs)
        Nouts = len(self.outputPro)
        pred = []
        refs = []
        for outPro in range(Nouts):
            pred.append([])
            refs.append([])
            for items in outputs:
                for item1, item2 in zip(items[f"pred{outPro}"], items[f"ref{outPro}"]):
                    pred[outPro].append(item1)
                    refs[outPro].append(item2)
            import matplotlib.pyplot as plt
            plt.scatter(refs, pred, marker="x", s=10)
            plt.xlabel("${ E_\mathrm{reference}}$/(eV)", size=20)
            plt.ylabel("${ E_\mathrm{prediction}}$/(eV)", size=20)
            print(F.mae_loss_for_metric(torch.Tensor(pred), torch.Tensor(refs)))
            plt.tight_layout()
            plt.show()
        np.savez("testresult", pred=pred, refs=refs)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        if getattr(self, "scheduler", None) is not None:
            return {"optimizer": self.optimizer,
                    'lr_scheduler': self.scheduler,
                    'monitor': 'val_0_loss_MAE',
                    'interval': 'epoch'
                    }
        else:
            return {"optimizer": self.optimizer,
                    'monitor': 'val_0_loss_MAE',
                    'interval': 'epoch'
                    }

    def todict(self):
        state_dict = self.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu().numpy()
        return state_dict
