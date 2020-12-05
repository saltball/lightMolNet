# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : functional.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import torch
import torch.nn.functional as F

mse_loss = F.mse_loss

mae_loss = F.smooth_l1_loss


def rsme_loss(input, target, reduction='mean'):
    return torch.sqrt(mse_loss(input, target, reduction))
