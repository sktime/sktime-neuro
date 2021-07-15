# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]
__all__ = ["TimeSlicingTransformer"]

from sktime_neuro.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X


class TimeSlicingTransformer(_PanelToPanelTransformer):
    """Slice data into region of interest

    Parameters
    _________
        start : float or None (default None),
            start of time frame of interest in seconds
        end : float or None (default None),
            end of time frame of interest in seconds
        fs : int or float
            sampling frequency of the recorded data in Hz
    """

    def __init__(self, start=None, end=None, fs=250):
        self.start = start
        self.end = end
        self.fs = fs
        super(TimeSlicingTransformer, self).__init__()

    def transform(self, X, y=None):
        """
        Slice data into region of interest.

        Parameters
        _________
        X : pd.DataFrame or np.array
        shape: trials*channels*timepoints

        Returns
        ________
        Xt : np.array
            truncated time series
        """
        if not (
            isinstance(self.start, (int, float, type(None)))
            & isinstance(self.end, (int, float, type(None)))
            & isinstance(self.fs, (int, float))
        ):
            raise TypeError(
                "start and end need to be numbers or none;" "fs needs to be a number."
            )

        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)

        # check if boundaries make sense
        if self.start is not None:
            lower_index = int(self.start * self.fs)
            if not (0 <= lower_index < X.shape[2]):
                raise ValueError("Lower limit is invalid, unit is seconds")
        else:
            lower_index = 0

        if self.end is not None:
            upper_index = int(self.end * self.fs)
            if not (0 <= upper_index < X.shape[2]):
                raise ValueError("Upper limit is invalid, unit is seconds")
        else:
            upper_index = X.shape[2]
        if self.start is not None and self.end is not None:
            if not (self.start < self.end):
                raise ValueError("Lower limit must be lower than upper limit")

        Xt = X[:, :, lower_index:upper_index]
        return Xt
