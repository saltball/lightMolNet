# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : partitioning.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import os

import numpy as np
from lightMolNet.logger import DebugLogger
from torch.utils.data import random_split

logger = DebugLogger(__name__)


def random_split_partial(
        data,
        partial=None,
        split_file="split"
):
    if partial is None:
        partial = [60, 20, 20]
    num_train = partial[0]
    num_val = partial[1]
    num_test = partial[2]
    if split_file is not None and os.path.exists(r"{}.npz".format(split_file)):
        S = np.load(r"{}.npz".format(split_file))
        train_idx = S["train_idx"].tolist()
        val_idx = S["val_idx"].tolist()
        test_idx = S["test_idx"].tolist()

    elif split_file is None or not os.path.exists(r"{}.npz".format(split_file)):
        if num_train is None or num_val is None:
            raise ValueError(
                "You have to supply either split sizes (num_train /"
                + " num_val) or an npz file with splits."
            )
        logger.debug(f"num_train = {num_train}; num_val={num_val}, len(data)={len(data)}")
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
    if split_file is not None and not os.path.exists(r"{}.npz".format(split_file)):
        np.savez(
            split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )
    train = data.create_subset(train_idx)
    val = data.create_subset(val_idx)
    test = data.create_subset(test_idx)
    return train, val, test
