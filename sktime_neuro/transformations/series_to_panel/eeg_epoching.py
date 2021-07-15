# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]
__all__ = ["Epoch"]


from sktime_neuro.transformations.base import _SeriesToPanelTransformer
import numpy as np
from sktime.utils.validation.series import check_series


class Epoch(_SeriesToPanelTransformer):
    """
    Epoch continuous EEG data into trials of label.
    """

    def __init__(self, annotation, label, interval, sfreq):
        self.annotation = annotation
        self.label = label
        self.interval = interval
        self.sfreq = sfreq
        super(Epoch).__init__()

    def transform(self, Z, y=None):
        """
        Parameters
        _________
        annotation : pd.DataFrame,
            one row per event with columns "onset", "duration"
            and "descritption"
            can be create from mne raw object
            with `sktime_neuro.utils.mne_processing.create_annotation`
        label : string
            label of event to create trials from
        interval : tuple of float or int
            time interval around event to select
        fs : int or float
            sampling frequency of the recorded data in Hz

        Returns
        ________
        Xt : np.array
             panel data with trials of event "label"
        """
        Z = check_series(Z)
        all_onsets = self.annotation.loc[lambda df: df["description"] == self.label][
            "onset"
        ]
        n_trials = len(all_onsets)
        n_channels = Z.shape[0]
        n_timepoints = int(self.interval[1] * self.sfreq) - int(
            self.interval[0] * self.sfreq
        )
        Xt = np.zeros((n_trials, n_channels, n_timepoints))
        for idx, onset in enumerate(all_onsets):
            offset = int(self.sfreq * onset)
            Xt[idx] = Z[
                :,
                (int(self.interval[0] * self.sfreq) + offset) : (
                    int(self.interval[1] * self.sfreq) + offset
                ),
            ]
        return Xt
