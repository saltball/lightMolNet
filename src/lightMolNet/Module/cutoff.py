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
           # "MollifierCutoff",
           # "HardCutoff",
           # "get_cutoff_by_string"
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
