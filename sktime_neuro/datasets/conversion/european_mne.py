import itertools
import os
import textwrap

import mne.io
from typing import List, Dict
from sktime.datasets._single_problem_loaders import load_UCR_UEA_dataset
import pandas as pd
import numpy as np

def write_jaggeddf_to_tsfile(
    data,
    path,
    problem_name="sample_data",
    class_label=None,
    class_value_list=None,
    equal_length=False,
    series_length=-1,
    missing_values="NaN",
    comment=None,
    fold="",
):
    """
    Output a dataset in jagged dataframe format to .ts file.

    Parameters
    ----------
    data: Jagged 3d List
        The list corresponding to the structure found here examples/loading_data.ipynb.
    path: str
        The full path to output the ts file to.
    problem_name: str, default="sample_data"
        The problemName to print in the header of the ts file and also the name of
        the file.
    class_label: list of str or None, default=None
        The problems class labels to show the possible class values for in the file
        header, optional.
    class_value_list: pd.series, ndarray or None, default=None
        The class values for each case, optional.
    equal_length: bool, default=False
        Indicates whether each series is of equal length.
    series_length: int, default=-1
        Indicates the series length if they are of equal length.
    missing_values: str, default="NaN"
        Representation for missing values.
    comment: str or None, default=None
        Comment text to be inserted before the header in a block.
    fold: str or None, default=None
        Addon at the end of the filename, i.e. _TRAIN or _TEST.

    Returns
    -------
    None

    Notes
    -----
    This is a hacked up version of the write_ndarray_to_tsfile from sktime's data_io, as that doesn't work with any
    uneqal length dataframes
    """


    univariate = False
    if class_value_list is not None and class_label is None:
        class_label = np.unique(class_value_list)
    elif class_value_list is None:
        class_value_list = []
    # ensure number of cases is same as the class value list
    if len(data) != len(class_value_list) and len(class_value_list) > 0:
        raise IndexError(
            "The number of cases is not the same as the number of given class values"
        )
    if equal_length and series_length == -1:
        raise ValueError(
            "Please specify the series length for equal length time series data."
        )
    if fold is None:
        fold = ""
    # create path if not exist
    dirt = f"{str(path)}/{str(problem_name)}/"
    try:
        os.makedirs(dirt)
    except os.error:
        pass  # raises os.error if path already exists
    # create ts file in the path
    file = open(f"{dirt}{str(problem_name)}{fold}.ts", "w")
    # write comment if any as a block at start of file
    if comment is not None:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")
    # begin writing header information
    file.write(f"@problemName {problem_name}\n")
    file.write("@timestamps false\n")
    file.write(f"@univariate {str(univariate).lower()}\n")
    file.write(f"@equalLength {str(equal_length).lower()}\n")
    if series_length > 0 and equal_length:
        file.write(f"@seriesLength {series_length}\n")
    # write class label line
    if class_label is not None:
        space_separated_class_label = " ".join(str(label) for label in class_label)
        file.write(f"@classLabel true {space_separated_class_label}\n")
    else:
        file.write("@class_label false\n")
    # begin writing the core data for each case
    # which are the series and the class value list if there is any
    file.write("@data\n")
    for case, value in itertools.zip_longest(data, class_value_list):
        for dimension in case:
            # turn series into comma-separated row
            series = ",".join(
                [str(num) if not np.isnan(num) else missing_values for num in dimension]
            )
            file.write(str(series))
            # continue with another dimension for multivariate case
            if not univariate:
                file.write(":")
        a = ":" if univariate else ""
        if value is not None:
            file.write(f"{a}{value}")  # write the case value if any
        elif class_label is not None:
            file.write(f"{a}{missing_values}")
        file.write("\n")  # open a new line
    file.close()

def loadAliceSleepData(): # If you only want to test on Alice
    return mne.datasets.sleep_physionet.age.fetch_data(subjects=[0], recording=[1], path="edfdatasets/")
    #Currently this is very brittle and will break if the datasets are unavaliable

def readRawAndSetAnnot(subject):
    """
    Extracts the raw data and applys the annotations
    Parameters
    ----------
    subject : Object
        Subject do extract raw from

    Returns
    -------
    Object
        MNE representation of raw data and annotations
    """
    raw = mne.io.read_raw_edf(subject[0], stim_channel="marker", misc=["rectal"])
    annotation = mne.read_annotations(subject[1])
    raw.set_annotations(annotation)
    return raw

def extractData(raw):
    channelList = []
    channelNames = []
    #Get the channel ids, the names and the data and put it in one nice list
    for i in range(len(raw.ch_names)):
        channelList.append({"ID" : i, "Name" : raw.ch_names[i], "Data" : raw[i]})
        channelNames.append(raw.ch_names[i])

    eventList = mne.events_from_annotations(raw)[0] #Times (in samples) and the class they belong to
    classLabels = list(mne.events_from_annotations(raw)[1].items()) # Names for the class labels
    classData = []
    classValues = []
    counter = 0
    #Iteerate though all channels and store (seperate each channel) until we reach a new event, in which create new array
    while counter < len(eventList)-1: #While we still got events to process, last event is always useless
        min = eventList[counter][0]
        if counter + 1 >= len(eventList):
            max = len(channelList[0]['Data'][0][0]) ## Total samples
        else:
            max = eventList[counter+1][0]
        tempArr = []
        for i in range(len(channelList)): #Iterate through the channels
            s = channelList[i]['Data'][0][0][min:max]
            tempArr.append(s)
        classData.append(tempArr) #Append to current working list
        classValues.append(eventList[counter][2])
        counter += 1
    return [classData, classLabels, classValues]

def conversionPipeline(subject, path="./", problem_name="Test"): #For now only take a subject
    raw = readRawAndSetAnnot(subject)
    [classData, classLabels, classValues] = extractData(raw)
    name = "test"
    pathToWrite = "./"
    write_jaggeddf_to_tsfile(classData, pathToWrite, problem_name=name, class_label=classLabels, class_value_list=pd.Series(classValues))

if __name__ == "__main__":
    conversionPipeline(loadAliceSleepData())
    testload = load_UCR_UEA_dataset("test")

    #Currently the inbuilt save function doesn't actually work for varying length!
    #write_dataframe_to_tsfile(insect[0], pathToWrite, problem_name=name, class_label=None, class_value_list=insect[1])