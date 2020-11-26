# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : statistics.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import logging

import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def get_statistics(
        dataloader: torch.utils.data.DataLoader,
        property_names: str or list,
        divide_by_atoms: dict or bool = False,
        single_atom_ref: dict or bool = None
):
    """
    Compute mean and variance of a property. Uses the incremental Welford
    algorithm implemented in StatisticsAccumulator

    Parameters
    ----------
        dataloader: torch.utils.data.DataLoader

        property_names:str or list
            Name of the property for which the
            mean and standard deviation should be computed
        divide_by_atoms:dict or bool
            divide mean by number of atoms if True
            (default: False)
        single_atom_ref:dict or bool
            reference values for single atoms
            (default: None)

    Returns
    -------
        mean:
            Mean value
        stddev:
            Standard deviation

    """
    if type(property_names) is not list:
        property_names = [property_names]
    if type(divide_by_atoms) is not dict:
        divide_by_atoms = {prop: divide_by_atoms for prop in property_names}
    if single_atom_ref is None:
        single_atom_ref = {prop: None for prop in property_names}

    with torch.no_grad():
        statistics = {
            prop: StatisticsAccumulator(batch=True) for prop in property_names
        }
        logger.info("statistics will be calculated...")

        for row in dataloader:
            for prop in property_names:
                _update_statistic(
                    divide_by_atoms[prop],
                    single_atom_ref[prop],
                    prop,
                    row,
                    statistics[prop],
                )

        means = {prop: s.get_mean() for prop, s in statistics.items()}
        stddevs = {prop: s.get_stddev() for prop, s in statistics.items()}

    return means, stddevs


def _update_statistic(
        divide_by_atoms,
        single_atom_ref,
        property_name,
        row,
        statistics
):
    """
        Helper function to update iterative mean / stddev statistics
    """
    property_value = row[property_name]
    if single_atom_ref is not None:
        z = row["_atomic_numbers"]
        p0 = torch.sum(torch.from_numpy(single_atom_ref[z]).float(), dim=1)
        property_value -= p0
    if divide_by_atoms:
        property_value /= torch.sum(row["_atom_mask"], dim=1, keepdim=True)
    statistics.add_sample(property_value)


class StatisticsAccumulator:
    """
    Attributes
    ----------
        count: int
        mean: torch.Tensor
        M2: torch.Tensor
    """

    def __init__(self, batch: bool = False, atomistic: bool = False):
        """
        Use the incremental Welford algorithm described in [1]_ to accumulate
        the mean and standard deviation over a set of samples.

        Parameters
        ----------
            batch:
                If set to true, assumes sample is batch and uses leading
                   dimension as batch size
            atomistic:
                If set to true, average over atom dimension



        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        """
        # Initialize state variables
        self.count = 0  # Sample count
        self.mean = 0  # Incremental average
        self.M2 = 0  # Sum of squares of differences
        self.batch = batch
        self.atomistic = atomistic

    def add_sample(self, sample_value):
        """
        Add a sample to the accumulator and update running estimators.
        Differentiates between different types of samples.

        Parameters
        ----------
            sample_value:torch.Tensor
                data sample
        """

        # Check different cases
        if not self.batch and not self.atomistic:
            self._add_sample(sample_value)
        elif not self.batch and self.atomistic:
            n_atoms = sample_value.size(0)
            for i in range(n_atoms):
                self._add_sample(sample_value[i, :])
        elif self.batch and not self.atomistic:
            n_batch = sample_value.size(0)
            for i in range(n_batch):
                self._add_sample(sample_value[i, :])
        else:
            n_batch = sample_value.shape[0]
            n_atoms = sample_value.shape[1]
            for i in range(n_batch):
                for j in range(n_atoms):
                    self._add_sample(sample_value[i, j, :])

    def _add_sample(self, sample_value):
        # Update count
        self.count += 1
        delta_old = sample_value - self.mean
        # Difference to old mean
        self.mean += delta_old / self.count
        # Update mean estimate
        delta_new = sample_value - self.mean
        # Update sum of differences
        self.M2 += delta_old * delta_new

    def get_statistics(self):
        """
        Compute statistics of all data collected by the accumulator.

        Returns
        -------
            tuple(torch.Tensor,torch.Tensor):
                Mean of data, Standard deviation of data
        """
        # Compute standard deviation from M2
        mean = self.mean
        stddev = torch.sqrt(self.M2 / self.count)

        return mean, stddev

    def get_mean(self):
        return self.mean

    def get_stddev(self):
        return torch.sqrt(self.M2 / self.count)
