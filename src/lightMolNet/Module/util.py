# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : util.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

from functools import partial

import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter

zeros_initializer = partial(constant_, val=0.0)


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

        .. math::
           y = F_{act}(xW^T + b)

        Parameters
        ----------
            in_features:int
                number of input feature :math:`x`.
            out_features:int
                number of output features :math:`y`.
            bias:bool, optional
                if False, the layer will not adapt bias :math:`b`.
            activation:callable, optional
                if None, no activation function is used.
            weight_init:callable, optional
                weight initializer from current weight.
            bias_init:callable, optional
                bias initializer from current bias.

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation: callable = None,
            weight_init: callable = xavier_uniform_,
            bias_init: callable = zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation

        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """Reinitialize Dense Layers' weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Parameters
        ----
            inputs: # dict of torch.Tensor
                batch of input values.

        Returns
        -------
            torch.Tensor:
                layer output

        """
        y = super(Dense, self).forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y


class Aggregate(nn.Module):
    """Pooling layer based on sum or average with optional masking.

    Args
    ----
        axis:int
            axis along which pooling is done.
        mean:bool, optional
            if True, use average instead for sum pooling.
        keepdim:bool, optional
            whether the output tensor has dim retained or not.

    """

    def __init__(
            self,
            axis: int,
            mean: bool = False,
            keepdim: bool = True
    ):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def forward(
            self,
            input: torch.Tensor,
            mask: torch.Tensor = None):
        r"""Compute layer output.

        Parameters
        ----------
            input:torch.Tensor
                input data.
            mask:torch.Tensor, optional
                mask to be applied; e.g. neighbors mask.

        Returns
        -------
            torch.Tensor:
                layer output.

        """
        # mask input
        if mask is not None:
            input = input * mask[..., None]
        # compute sum of input along axis
        y = torch.sum(input, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N
        return y


class GetItem(nn.Module):
    r"""Extraction layer to get an item from dictionary of input tensors.

    Args
    ----
        key:str
            Property to be extracted from input tensors.

    """

    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        r"""Compute layer output.

        Args
        ----
            inputs:dict of str:torch.Tensor
                dictionary of input tensors.

        Returns
        -------
            torch.Tensor:
                layer output.

        """
        return inputs[self.key]


class ScaleShift(nn.Module):
    r"""Scale and shift layer for standardization.

    .. math::
       y = x \times \sigma + \mu

    Parameters
    ----------
        mean:torch.Tensor
            mean value :math:`\mu`.
        stddev:torch.Tensor
            standard deviation value :math:`\sigma`.
        learnable:bool
            if True, mean and stddev are parameters to learn.

    """

    def __init__(self, mean=(0.), stddev=(1.), learnable=True):
        super(ScaleShift, self).__init__()
        if not learnable:
            self.register_buffer("mean", mean)
            self.register_buffer("stddev", stddev)
        else:
            self.mean = Parameter(torch.Tensor([mean]))
            self.stddev = Parameter(torch.Tensor([stddev]))

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y
