# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]

import numpy as np
from sktime.utils.validation.series import check_series


def epoch(Z, annotation, labels, interval, sfreq) -> (np.array, np.array):
    """
    Parameters
    _________
    Z : np.array
        time series to be epoched
        shape: timepoints*channels
    annotation : pd.DataFrame,
        one row per event with columns "onset", "duration"
        and "descritption"
        can be create from mne raw object
        with `sktime_neuro.utils.mne_processing.create_annotation`
    labels : list of string
        labels of events to create trials from
    interval : tuple of float or int
        time interval around event to select
    fs : int or float
        sampling frequency of the recorded data in Hz

    Returns
    ________
    Xt : np.array
        panel data (shape: trials, channels, timepoints)
    y : np.array
        labels vector
    """

    Z = check_series(Z)

    # create shape of final data
    n_channels = Z.shape[1]
    n_timepoints = int(interval[1] * sfreq) - int(interval[0] * sfreq)
    n_trials = len(annotation.loc[lambda df: df["description"].isin(labels)]["onset"])
    if n_trials == 0:
        raise ValueError(
            "Data does not contain trials that "
            "correspond to any of the provided labels."
        )
    X = np.zeros((n_trials, n_channels, n_timepoints))
    y = []
    idx = 0
    # iterate over annotator and add data parts that belong to label
    for _, row in annotation.iterrows():
        if row["description"] in labels:
            offset = int(sfreq * row["onset"])
            X[idx] = (
                Z[
                    (int(interval[0] * sfreq) + offset) : (
                        int(interval[1] * sfreq) + offset
                    ),
                    :,
                ]
            ).transpose()
            y.append(row["description"])
            idx += 1

    return X, np.asarray(y)
