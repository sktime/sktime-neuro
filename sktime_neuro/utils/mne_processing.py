# -*- coding: utf-8 -*-
__author__ = ["Svea Marie Meyer"]

import pandas as pd
import numpy as np
import mne
from sktime.utils.validation.series import check_series


def create_annotation(raw):
    """
    Create sktime_neuro annotation from raw-mne object.

    Parameters
    _________
    raw : mne raw object

    Returns
    ________
    annotation_pandas : pd.DataFrame
        one row per event with columns
        "onset", "duration" and "descritption"

    """
    annotation_pandas = pd.DataFrame(columns=["onset", "duration", "description"])
    for idx, event in enumerate(raw.annotations):
        annotation_pandas.loc[idx] = [
            event["onset"],
            event["duration"],
            event["description"],
        ]
    return annotation_pandas


def create_mne_raw(series, s_freq, ch_names=None, highpassed=True, **kwargs):
    """
    Create mne raw object from a series.

    Parameters
    _________
    series : pd.Series, pd.DataFrame, np.array
        recorded data
    s_freq : int or float
        sampling frequency of the recorded data in Hz
    ch_names : List of strings
        names of the channels
    highpassed : bool
        indicates whether data was already highpassed filtered

    **kwargs : kwargs
        Additional parameters passed on to ``mne.create_info``.
        See ``mne.create_info``
        documentation for a detailed description of all options.

    Returns
    ________
    mne_raw : mne raw object
    """

    series = check_series(series)
    if isinstance(series, pd.DataFrame):
        if ch_names is None:
            ch_names = list(series.columns)
        ch_types = ["eeg"] * len(ch_names)
        data = series[ch_names].to_numpy()

    elif isinstance(series, np.ndarray):
        ch_types = ["eeg"] * series.shape[0]
        if ch_names is None:
            integer_names = list(range(series.shape[0]))
            ch_names = [str(x) for x in integer_names]
        data = series

    else:  # its a pd.Series
        ch_types = ["eeg"]
        if series.ch_names is None:
            series.ch_names = ["0"]
        data = series

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=s_freq, **kwargs)
    mne_raw = mne.io.RawArray(data, info)
    if highpassed:
        mne_raw.info["highpass"] = 1
    return mne_raw
