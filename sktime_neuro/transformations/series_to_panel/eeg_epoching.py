# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]

import numpy as np
from sktime.utils.validation.series import check_series


def epoch(Z, annotation, labels, interval, sfreq) -> (np.array, np.array):
    """
    Parameters
    _________
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
    n_channels = Z.shape[0]
    n_timepoints = int(interval[1] * sfreq) - int(interval[0] * sfreq)
    n_trials = len(annotation.loc[lambda df: df["description"].isin(labels)]["onset"])
    X = np.zeros((n_trials, n_channels, n_timepoints))
    y = []

    # iterate over specified labels and get corresponding trials
    idx = 0
    for label in labels:

        # get all onsets that correspond to the label we are currently looking at
        onsets_of_label = annotation.loc[lambda df: df["description"] == label]["onset"]

        # iterate over onsets and add them to the datacontainer (X) and labels (y)
        for onset in onsets_of_label:
            offset = int(sfreq * onset)
            X[idx] = Z[
                :,
                (int(interval[0] * sfreq) + offset) : (
                    int(interval[1] * sfreq) + offset
                ),
            ]
            y.append(label)
            idx += 1
    return X, np.asarray(y)
