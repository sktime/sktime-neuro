# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]
__all__ = ["FilterforSeries"]

import pandas as pd
from sktime_neuro.transformations.base import _SeriesToSeriesTransformer
from mne import filter
from sktime.utils.validation.series import check_series
import numpy as np

_required_parameter = ["sfreq", "l_freq", "h_freq"]


class FilterforSeries(_SeriesToSeriesTransformer):
    """Transformer that filters Series data.

    Provides a simple wrapper around ``mne.filter.filter_data``.

    Parameters
    ----------
    fs: int or float
        sampling frequency of the recorded data in Hz
    l_freq: float or None
        For FIR filters, the lower pass-band edge;
        for IIR filters, the lower cutoff frequency.
        If None the data are only low-passed.
    h_freq: float or None
        For FIR filters, the upper pass-band edge;
        for IIR filters, the upper cutoff frequency.
        If None the data are only high-passed.
    **kwargs
        Additional parameters passed on to ``mne.filter.filter_data``.
        See ``mne.filter.filter_data``
        documentation for a detailed description of all options.
    """

    def __init__(
        self,
        sfreq,
        l_freq,
        h_freq,
        **kwargs,
    ):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.kwargs = kwargs
        if not (
            isinstance(sfreq, (int, float))
            & isinstance(l_freq, (int, float, type(None)))
            & isinstance(h_freq, (int, float, type(None)))
        ):
            raise TypeError
        elif (l_freq is not None) & (h_freq is not None):
            if not ((l_freq > 0) & (h_freq > 0)):
                raise ValueError("Negative values not supported")
            if l_freq > h_freq:
                raise ValueError("High frequency must be higher" " than low frequency")
        super(FilterforSeries, self).__init__()

    def transform(self, Z, x=None) -> np.array:
        """Transform data.
        Returns a transformed version of Z.

        Parameters
        ----------
        Z : 2D numpy array, pd.Series or pd.DataFrame
        (will get coerced to numpy)

        Returns
        -------
        z : 2D numpy array
            Transformed time series.
        """
        self.check_is_fitted()
        z = check_series(Z, allow_numpy=True)

        # mne only deals with numpy arrays:
        if isinstance(z, (pd.DataFrame, pd.Series)):
            z = z.to_numpy()

        z = filter.filter_data(
            z, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, **self.kwargs
        )

        return z
