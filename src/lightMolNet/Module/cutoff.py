# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : cutoff.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch
from torch import nn

__all__ = ["CosineCutoff",
           "RBFCutoff"
           ]


class CosineCutoff(nn.Module):
    r"""CosineCutoff function

    Cutoff Layer using Cosine function for distances tensor.

    Parameters
    ----------
    cutoff : float
        Cutoff offset value of the Layer.

    Notes
    -----
    The cosine cutoff return distance value :math:`d^'` transferred by cosine function:

    .. math:: d'=\frac{1}{2}\left(\cos(\frac{d}{d_{cut}})+1\right)
    """

    def __init__(self, cutoff: float = 5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distance: torch.Tensor):
        r"""Forward broadcast by cutoff.



        Parameters
        ----------
        distance : torch.Tensor
            Values of input, especially distances data.

        Returns
        -------
        torch.Tensor
            Values of cutoff-ed distance data.

        """

        result = 0.5 * (torch.cos(distance * np.pi / self.cutoff) + 1.0)
        result *= (result < self.cutoff).float()
        return result


class RBFCutoff(nn.Module):
    r"""RBFCutoff function Layer

    Cutoff Layer using radical basis functions for distances tensor.

    Parameters
    ----------
    n_rbf_basis: int
        Number of radical basis functions used in this layer.
    cutoff : float
        Cutoff offset value of the Layer.

    Notes
    -----
    The rbfcutoff return expanded values from distance
    :math:`rbf_{out}` expanded an exponential function to distance on a set of gaussian function:

    .. math:: `\{g_k\}(r)`
    In which :math:`\Phi(r)=1-6r^5+15r^4-10r^3,r=r_{ij}/r_{cut}`,and
            :math:`g_k(r)=\Phi(r)\times \exp{(-\beta_k(\exp{(-r_{ij})}-\mu_k)^2)}`
    """

    def __init__(self, n_rbf_basis: int, cutoff: float):
        super().__init__()
        self._n_rbf_basis = n_rbf_basis
        self._cutoff = cutoff
        centers = torch.linspace(1.0, np.exp(-self.cutoff), self.n_rbf_basis)
        width = torch.FloatTensor((0.5 / ((1.0 - np.exp(-self.cutoff)) / self.n_rbf_basis),)) ** 2
        self.width = nn.Parameter(width)  # (1,)
        self.centers = nn.Parameter(centers)  # (n_rbf_basis,)

    def forward(self, D):
        # D (nbatch,nat,nbh)
        D = D[:, :, :, None]
        Dr = D / self.cutoff
        D3 = Dr ** 3
        D4 = D3 * Dr
        D5 = D4 * Dr
        phi = torch.where(Dr < 1,
                          1 - 6 * D5 + 15 * D4 - 10 * D3,
                          torch.zeros_like(Dr))
        return phi * torch.exp(-self.width * (torch.exp(-D) - self.centers) ** 2)

    @property
    def n_rbf_basis(self):
        return self._n_rbf_basis

    @property
    def cutoff(self):
        return self._cutoff
