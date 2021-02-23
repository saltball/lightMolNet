# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : residual.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from torch import nn
from torch.nn.init import orthogonal_

from lightMolNet.Module.util import Dense


class ResidualLayer(nn.Module):
    r"""Residual layer with activation function.

            .. math::
               x^{l+2} = x^{l}+W^{l+1}F_{act}(W^{l}F_{act}x^{l} + b^{l})+b^{l+1}

            Parameters
            ----------
                in_features:int
                    number of input feature :math:`x_0`.
                out_features:int
                    number of output features :math:`x_0+\sum x_{res}`.
                bias:bool, optional
                    if False, the layer will not adapt bias :math:`b`.
                activation:callable, optional
                    if None, no activation function is used.
        """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: callable = None,
    ):
        super(ResidualLayer, self).__init__()
        self.dense = Dense(in_features, out_features, activation=activation, bias=bias, weight_init=orthogonal_)
        self.residual = Dense(out_features, out_features, activation=activation, bias=bias, weight_init=orthogonal_)
        self.activation = activation

    def forward(self, x):
        # pre-activation
        if self.activation is not None:
            y = self.activation(x)
        else:
            y = x
        x = x + self.residual(self.dense(y))
        return x
