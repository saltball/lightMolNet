# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : __init__.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import ase.calculators.calculator as ase_calc
import numpy as np
import torch
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError
from ase.units import Angstrom
from ase.units import Hartree, eV

from lightMolNet import InputPropertiesList
from lightMolNet.data.atoms2input import convert_atoms


class torchCaculator(Calculator):
    implemented_properties = [
        "energy",
        "forces"
    ]

    default_parameters = {
        "method": "SchNet",
        "accuracy": 1.0,
        "max_iterations": 250,
        "solvent": "None",
        "cache_api": True,
    }

    def __init__(
            self, atoms: [Atoms] = None, use_gpu=False, **kwargs,
    ):
        """Construct the xtb base calculator object."""

        self.net: torch.nn.Module
        super(torchCaculator, self).__init__(atoms=atoms, **kwargs)

        self.net = self.parameters.net
        if not hasattr(self.parameters, "net"):
            raise CalculatorSetupError("Please implement the `net` parameter!")
        self.use_gpu = use_gpu
        self.model = {"energy": None, "gradient": None}

    def calculate(
            self,
            atoms=None,
            properties=None,
            system_changes=ase_calc.all_changes,
    ):
        if not properties:
            properties = ["energy"]
        ase_calc.Calculator.calculate(self, atoms, properties, system_changes)

        result, ref = convert_atoms(atoms)
        result[InputPropertiesList.idx] = torch.LongTensor(np.array([1], dtype=np.int))
        for idx, k in enumerate(result):
            if k is not None:
                if self.use_gpu:
                    result[idx] = k[None, :].to(device="cuda")
                else:
                    result[idx] = k[None, :]

        self._model(result)

        self.results["energy"] = self.model["energy"][0] * eV / Hartree
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -self.model["gradient"] * eV / Hartree / Angstrom

    def _model(self, batch):
        inputs = batch
        inputs[InputPropertiesList.R].requires_grad = True
        self.net.freeze()
        if self.use_gpu:
            self.net.cuda()
        else:
            pass
        energy = self.net(inputs)
        if isinstance(energy, tuple):
            [item.backward() for item in energy[0]]
        else:
            [item.backward() for item in energy.values()]
        forces = inputs[InputPropertiesList.R].grad
        if not isinstance(energy, dict):
            self.model["energy"] = energy.detach().cpu().numpy()[0]
        else:
            self.model["energy"] = energy["energy_U0"].detach().cpu().numpy()[0]
        self.model["gradient"] = forces.cpu().numpy()[0]
