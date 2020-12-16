# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : atomcentersymfunc.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import torch
from torch import nn


def gaussian_smearing(
        distances: torch.Tensor,
        offset: torch.Tensor,
        widths: torch.Tensor,
        centered: bool = False
):
    r"""Smear interatomic distance values using Gaussian functions.

    Parameters
    ----------
        distances : torch.Tensor
            interatomic distances of (N_b x N_at x N_nbh) shape.
        offset: torch.Tensor
            offsets values of Gaussian functions.
        widths : torch.Tensor
            width values of Gaussian functions.
        centered : bool, optional
            If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).

    Returns
    -------
        torch.Tensor:
            smeared distances (N_b x N_at x N_nbh x N_g).

    Notes
    -----
        gaussian_smearing return the values like

        .. math:: exp(-0.5(1/w\Delta d)^2)
        And :math:`\Delta d` defined by whether the parameter `centered` is `True`.

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 * torch.pow(widths, -2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 * torch.pow(offset, -2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    r"""

    """

    def __init__(
            self,
            start: float = 0.0,
            stop: float = 5.0,
            n_gaussians: int = 50,
            centered: bool = False,
            trainable: bool = False
    ):
        super(GaussianSmearing, self).__init__()
        # offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(
            self,
            distances: torch.Tensor
    ):
        """

        Parameters
        ----------
        distances:torch.Tensor
            input Tensor

        Returns
        -------
            torch.Tensor:
                        Gaussian Smearing values.

        """
        return gaussian_smearing(
            distances,
            self.offsets,
            self.width,
            centered=self.centered
        )
