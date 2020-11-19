# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : partitioning.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import os

import numpy as np
from torch.utils.data import random_split


def random_split_partial(
        data,
        partial=None,
        split_file=None
):
    if partial is None:
        partial = [60, 20, 20]
    num_train = partial[0]
    num_val = partial[1]
    num_test = partial[2]
    if split_file is not None and os.path.exists(split_file):
        S = np.load(split_file)
        train_idx = S["train_idx"].tolist()
        val_idx = S["val_idx"].tolist()
        test_idx = S["test_idx"].tolist()

    else:
        if num_train is None or num_val is None:
            raise ValueError(
                "You have to supply either split sizes (num_train /"
                + " num_val) or an npz file with splits."
            )

        assert num_train + num_val <= len(
            data
        ), "Dataset is smaller than num_train + num_val!"

        n = len(data)
        assert n > 100, "too small dataset"
        partial = np.array(partial)
        partial = partial / partial.sum() * n
        n_train = int(partial[0])
        n_val = int(partial[1])
        n_test = n - n_train - n_val
        train_idx, val_idx, test_idx = random_split(
            range(n),
            [
                n_train, n_val, n_test
            ])
        train = data.create_subset(train_idx)
        val = data.create_subset(val_idx)
        test = data.create_subset(test_idx)
    return train, val, test
