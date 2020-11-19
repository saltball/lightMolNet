# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : activations.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import numpy as np
import torch
from torch.nn import functional


def shifted_softplus(
        x: torch.Tensor,
        beta: float = 1,
        shift: float = 2.0):
    r"""Compute shifted soft-plus activation function.

    .. math::
       \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))-\log(\Delta_{shift})

    Parameters
    ----------

    x: torch.Tensor
        input
    beta: torch.Tensor
        :math:`\beta` value in `torch.nn.functional.softplus`
    shift: float
        shift value

    Returns
    -------
    torch.Tensor:
        shifted soft-plus of input.

    See Also
    --------
    torch.nn.functional.softplus
    """

    return functional.softplus(x, beta=beta) - np.log(shift)
