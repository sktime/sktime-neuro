# -*- coding: utf-8 -*-
import scipy.io
import mne
from sktime_neuro.utils import mne_processing as utils
import pandas as pd
import numpy as np


def load_c4_ds1(path, subject, load="train"):
    """Load Data from the first dataset of the 4th BCI competition.

    Data to be found at: http://www.bbci.de/competition/iv/#datasets

    Parameters
    ----------

    path : str
        path to datasets
    subject : str
        number of subject to load
    load : str (either "train" or "test")
        laod train or test data

    Returns
    -------

    fs: float in Hz
        sampling frequency od the recorded data
    data : np.array
        recorded data with shape timepoints*channels
    annotation : pd.Dataframe
         one row per event with columns
        "onset", "duration" and "description"
    """

    # naming of subject is a to g instead of 1 to 7
    subject_mapping = {
        "01": "a",
        "02": "b",
        "03": "c",
        "04": "d",
        "05": "e",
        "06": "f",
        "07": "g",
    }

    # load corresponding matfile
    if load == "train":
        fname = path + "BCICIV_calib_ds1" + subject_mapping[subject] + ".mat"
    elif load == "test":
        fname = path + "BCICIV_eval_ds1" + subject_mapping[subject] + ".mat"
    else:
        raise ValueError("can only load test or train")

    m = scipy.io.loadmat(fname, struct_as_record=True)

    # get raw data
    data = m["cnt"].astype(float)

    # get sampling frequency
    fs = int(m["nfo"]["fs"][0][0][0][0])

    # create annotation
    # multiplying by fs to get onset time and not index
    event_onsets = m["mrk"][0][0][0][0] * (1 / fs)
    event_description = m["mrk"][0][0][1][0]
    annotation_pandas = pd.DataFrame(columns=["onset", "duration", "description"])
    annotation_pandas["onset"] = event_onsets
    annotation_pandas["description"] = event_description

    # not used so far:
    # channel_names = [s[0] for s in m["nfo"]["clab"][0][0][0]]
    return fs, data, annotation_pandas


def load_c4_ds2b(path, subject, load="train"):
    """Load Data from the first dataset of the 4th BCI competition.

    Data to be found at: http://www.bbci.de/competition/iv/#datasets

    Parameters
    ----------

    path : str
        path to datasets
    subject : str
        number of subject to load
    load : str (either "train" or "test")
        laod train or test data

    Returns
    -------

    fs: float in Hz
        sampling frequency od the recorded data
    data : np.array
        recorded data with shape timepoints*channels
    annotation : pd.Dataframe
         one row per event with columns
        "onset", "duration" and "description"
    """

    if load == "train":
        for i in range(1, 4):
            # 1 to 3 is training and 4 & 5 is testing
            # first load data into mne format
            fname = path + "B" + subject + "0" + str(i) + "T.gdf"
            if i == 1:
                raw = mne.io.read_raw_gdf(fname, preload=True)
            elif 1 < i < 4:
                raw_new = mne.io.read_raw_gdf(fname, preload=True)
                raw.append(raw_new)

    elif load == "test":
        # 1 to 3 is training and 4 & 5 is testing
        # first load data into mne format
        for i in range(4, 6):
            fname = path + "B" + subject + "0" + str(i) + "E.gdf"
            if i == 4:
                raw = mne.io.read_raw_gdf(fname)
            elif i == 5:
                raw_new = mne.io.read_raw_gdf(fname)
                raw.append(raw_new)

    # create data and annotation from mne object
    annotation = utils.create_annotation(raw)

    # get numpy array from mne data, only include eeg channels
    # and transpose to achieve shape timepoints*channels
    data = raw.pick_types(eeg=True).get_data().transpose()

    fs = raw.info["sfreq"]

    return fs, data, annotation


def load_BNCI_2(path, subject, load="train"):
    """Load Data from the second dataset of BNCI Horizon 2020.

    Data to be found at: http://bnci-horizon-2020.eu/database/data-sets

    Parameters
    ----------

    path : str
        path to datasets
    subject : str
        number of subject to load
    load : str (either "train" or "test")
        laod train or test data

    Returns
    -------

    fs: float in Hz
        sampling frequency od the recorded data
    data : np.array
        recorded data with shape timepoints*channels
    annotation : pd.Dataframe
         one row per event with columns
        "onset", "duration" and "description"
    """
    if load == "train":
        fname = path + "S" + subject + "T.mat"
        runs = 5
    elif load == "test":
        fname = path + "S" + subject + "E.mat"
        runs = 3
    else:
        raise ValueError("can only load test or train")

    m = scipy.io.loadmat(fname, struct_as_record=True)
    annotation_pandas = pd.DataFrame(columns=["onset", "duration", "description"])

    fs = 512
    len_of_last_run = 0
    for run in range(runs):
        if run == 0:
            data = m["data"][0][run][0][0][0]
            onsets = ((m["data"][0][run][0][0][1] + len_of_last_run) * 1 / fs).flatten()
            labels = (m["data"][0][run][0][0][2]).flatten()
            len_of_last_run = data.shape[0]
        else:
            new_data = m["data"][0][run][0][0][0]
            data = np.vstack((data, new_data))
            new_onsets = (
                (m["data"][0][run][0][0][1] + len_of_last_run) * 1 / fs
            ).flatten()
            onsets = np.concatenate((onsets, new_onsets))
            new_labels = (m["data"][0][run][0][0][2]).flatten()
            labels = np.concatenate((labels, new_labels))
            len_of_last_run = data.shape[0]

    annotation_pandas["onset"] = onsets
    annotation_pandas["description"] = labels

    return fs, data, annotation_pandas


if __name__ == "__main__":
    competition = "c4_ds1"

    if competition == "c4_ds2b":
        path = "Data/c4_ds2b/"
        fs, data, annotation = load_c4_ds2b(path=path, subject="04")
    elif competition == "c4_ds1":
        path = "Data/c4_ds1/"
        fs, data, annotation = load_c4_ds1(path=path, subject="04")
    elif competition == "bnci_2":
        path = "Data/bnci_2/"
        fs, data, annotation = load_BNCI_2(path=path, subject="01")
    else:
        raise ValueError("Dataset loader not available")
