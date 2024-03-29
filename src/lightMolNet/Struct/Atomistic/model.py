# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : model.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import torch
from lightMolNet import Properties
from torch import nn


class AtomisticModel(nn.Module):
    r"""
    Join a representation model with output modules.

    Args
    ----
        representation:torch.nn.Module
            Representation block of the model.
        output_modules:list or nn.ModuleList or lightMolNet.
            Output block of the model. Needed for predicting properties.

    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])
        # For stress tensor
        self.requires_stress = any([om.stress for om in self.output_modules])

    def forward(self, inputs):
        r"""

        Parameters
        ----------
        inputs: dict of str:torch.Tensor


        Returns
        -------
        outs: dict of str:torch.Tensor

        """
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            # Generate Cartesian displacement tensor
            displacement = torch.zeros_like(inputs[Properties.cell]).to(
                inputs[Properties.R].device
            )
            displacement.requires_grad = True
            inputs["displacement"] = displacement

            # Apply to coordinates and cell
            inputs[Properties.R] = inputs[Properties.R] + torch.matmul(
                inputs[Properties.R], displacement
            )
            inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
                inputs[Properties.cell], displacement
            )

        inputs["representation"] = self.representation(inputs)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs
