"""
Handlers for dealing with EDF and EDF+ files.
Both of these formats are 16-bit.
"""
# See https://github.com/holgern/pyedflib and
# https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html

from pyedflib import highlevel
import numpy as np
import pandas as pd
from sktime.utils.data_io import *


def read_edf(filepath : str) -> np.ndarray:
    return highlevel.read_edf(filepath)


def extract_labels(sighead):
    labels = []
    for head in sighead:
        labels.append(head["label"])
    return labels


def extract_channels(sig):
    features = []
    for featureSet in sig:
        features.append(featureSet)
    return features


def extract_classes(annotations: list) -> dict:
    classNames = []
    for annotation in annotations:
        classNames.append(annotation[2])
    print("Wait")
    rv = list(set(classNames))
    rvd = dict(enumerate(rv, start=1))
    return rv


def split_channel_to_induvidual_observations(dataSignal, annotationHeader):
    observations = []
    for observationMeta in annotationHeader["annotations"]:
        start = int(observationMeta[0])
        end = int(start + observationMeta[1])
        obs = dataSignal[start:end]
        observations.append(obs)

    return observations


def extract_annotations(annotationHeader):
    annotations = []
    for annotation in annotationHeader["annotations"]:
        annot = annotation[2]
        annotations.append(annot)
    return annotations


def handle_multipart(signalFile, annotationFile, outputPath, problemName, isUniVariate):
    """
    In some cases, the time series will contain both a file which holds signals
    and one file that contains the annotations and metadata. As such we need
    to handle this case sepperatley.
    In regard to data that is publically available, this appears to be the
    most common format for data.
    """
    (dataSignals, dataMetadata, dataHeader) = read_edf(signalFile)

    # Extract each channel into a sepperate list.
    channels = extract_channels(dataSignals)
    channelNames = extract_labels(dataMetadata)

    (annotationSignals, annotationMetadata, annotationHeader) = read_edf(annotationFile)
    observationsPerChannel = []
    for channel in channels:
        ob = split_channel_to_induvidual_observations(channel, annotationHeader)
        observationsPerChannel.append(ob)
    ops = np.transpose(observationsPerChannel)
    df = pd.DataFrame(ops, columns=channelNames)
    annotationVec = extract_annotations(annotationHeader)
    write_dataframe_to_tsfile(df, outputPath, problemName, class_value_list=annotationVec, univariate=isUniVariate, equal_length=False)


if __name__ == "__main__":
    handle_multipart("/home/patchouli/temp/ecg/SC4001E0-PSG.edf", "/home/patchouli/temp/ecg/SC4001EC-Hypnogram.edf", "~/output.ts", "test", False)
    print("Wait")
