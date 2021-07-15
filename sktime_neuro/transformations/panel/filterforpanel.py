# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]
__all__ = ["FilterforPanel"]


from sktime_neuro.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X
from mne import filter

_required_parameter = ["sfreq", "l_freq", "h_freq"]


class FilterforPanel(_PanelToPanelTransformer):
    """Transformer that filters Panel data.

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
                raise ValueError("High frequency must be higher /" "than low frequency")
        super(FilterforPanel, self).__init__()

    def transform(self, Z, x=None):
        """Transform data.
        Returns a transformed version of Z.

        Parameters
        ----------
        Z : pd.Series/np.array
            shape needs to have timepoints as last dimension

        Returns
        -------
        z : np.array
            Transformed time series.
        """

        self.check_is_fitted()
        Z = check_X(Z, coerce_to_numpy=True)
        z = filter.filter_data(
            Z, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, **self.kwargs
        )

        return z
