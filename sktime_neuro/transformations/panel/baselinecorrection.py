# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]
__all__ = ["BaselineCorrectionTransformer"]


from sktime_neuro.transformations.base import _PanelToPanelTransformer
import numpy as np
from sktime.utils.validation.panel import check_X


class BaselineCorrectionTransformer(_PanelToPanelTransformer):
    """Apply trialwise baseline correction on each channel.

    Parameters
    ----------
    lower : int or None (default None)
            lower limit of baseline segment in seconds

    upper : int or None (default None)
            upper limit of baseline segment
    fs : int or float
        sampling frequency of the recorded data in Hz

    If start and end are None the trial gets normalized: average is calculated
    over entire trial per channel and subtracted from each timepoint in that
    channel.
    """

    def __init__(self, lower=None, upper=None, fs=2):
        self.lower = lower
        self.upper = upper
        self.fs = fs
        super(BaselineCorrectionTransformer).__init__()

    def transform(self, X, y=None) -> np.array:
        """
        Transform X by averaging over interval i and subtracting that value
        from entire trial.

        Parameters
        _________
        X : pd.DataFrame or np.array,
            shape: trials*channels*timepoints

        Returns
        ________
        Xt : np.array
             baseline corrected panel data
        """
        if not (
            isinstance(self.lower, (int, float, type(None)))
            & isinstance(self.upper, (int, float, type(None)))
            & isinstance(self.fs, (int, float))
        ):
            raise TypeError(
                "start and end need to be numbers or none;" "fs needs to be a number."
            )
        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)

        # check if boundaries make sense
        if self.lower is not None:
            lower_index = int(self.lower * self.fs)
            if not (0 <= lower_index < X.shape[2]):
                raise ValueError("Lower limit is invalid, unit is seconds")
        else:
            lower_index = 0

        if self.upper is not None:
            upper_index = int(self.upper * self.fs)
            if not (0 <= upper_index < X.shape[2]):
                raise ValueError("Upper limit is invalid, unit is seconds")
        else:
            upper_index = X.shape[2]
        if self.lower is not None and self.upper is not None:
            if not (self.lower < self.upper):
                raise ValueError("Lower limit must be lower than upper limit")

        # apply baseline correction
        Xt = np.zeros(X.shape)  # shape: trial*channel*timepoints
        baseline_means = np.mean(
            X[:, :, lower_index:upper_index], axis=2
        )  # shape trial*channel

        # for subtraction trailing axes need to be the same
        Xt = np.transpose(X, (2, 0, 1)) - baseline_means
        Xt = np.transpose(Xt, (1, 2, 0))

        # what the section above does in verbose
        # for trial in range(X.shape[0]):
        #   for channel in range(X.shape[1]):
        #       Xt[trial, channel, :] = X[trial,channel, :] -
        #       np.mean(X[trial, channel, self.lower:self.upper])

        return Xt
