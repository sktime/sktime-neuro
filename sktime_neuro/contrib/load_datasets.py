# -*- coding: utf-8 -*-
import scipy.io
import mne
from sktime_neuro.utils import mne_processing as utils
import pandas as pd


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
        "1": "a",
        "2": "b",
        "3": "c",
        "4": "d",
        "5": "e",
        "6": "f",
        "7": "g",
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
    data = m["cnt"]

    # get sampling frequency
    fs = m["nfo"]["fs"][0][0][0][0]

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
            fname = path + "B0" + subject + "0" + str(i) + "T.gdf"
            if i == 1:
                raw = mne.io.read_raw_gdf(fname, preload=True)
            elif 1 < i < 4:
                raw_new = mne.io.read_raw_gdf(fname, preload=True)
                raw.append(raw_new)

    elif load == "test":
        # 1 to 3 is training and 4 & 5 is testing
        # first load data into mne format
        for i in range(4, 6):
            fname = path + "B0" + subject + "0" + str(i) + "E.gdf"
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


def load_BCNI_2():
    pass


if __name__ == "__main__":
    competition = "c4_ds2b"

    if competition == "c4_ds2b":
        path = "Data/c4_ds2b/"
        fs, data, annotation = load_c4_ds2b(path=path, subject="4")
    elif competition == "c4_ds1":
        path = "Data/c4_ds1/"
        fs, data, annotation = load_c4_ds1(path=path, subject="4")
    else:
        raise ValueError("Dataset not available")
