# -*- coding: utf-8 -*-
__author__ = ["Svea Meyer"]
__all__ = ["PanelDownsampling"]

from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X


class PanelDownsampling(_PanelToPanelTransformer):
    """
    Downsample X by factor.

    Parameters
    _________
    factor : int
        downsampling factor
    fs : int or float
        sampling frequency of the recorded data in Hz

    Downsampling keeps only the data in position
    of a multiple of factor.
    So downsampling by a factor 1 keeps all the data,
    downsampling by 2 keeps half the
    data, etc.
    """

    def __init__(self, factor=2):
        self.factor = factor
        if not isinstance(self.factor, int):
            raise TypeError("Can only downsample by whole integers")
        super(PanelDownsampling, self).__init__()

    def transform(self, X, y=None):
        """
        Take every factorth element of a trial.

        Parameters
        _________
        X : pd.DataFrame or Numpy array
            shape: trials*channels*timepoints

        Returns
        ________
        Xt : Numpy Array,
            Downsampled time series

        """

        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)

        if self.factor > X.shape[2]:
            raise ValueError("Factor too high.")

        Xt = X[:, :, 0 :: self.factor]
        # do we need a warning about this?
        # print("The new sampling frequency is:" + str(self.fs / self.factor))
        return Xt
