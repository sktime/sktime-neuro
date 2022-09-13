# -*- coding: utf-8 -*-
import os


import sys
import time
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from sktime.benchmarking.experiments import load_and_run_classification_experiment
from sktime.datasets import write_results_to_uea_format
from sktime.datasets import load_UCR_UEA_dataset
from sktime.classification.feature_based import FreshPRINCE


os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!


"""Seven EEG/MEG classification problems currently in the
timeseriesclassification.com archive"""
eeg_problems = [
    "EyesOpenShut",
    "FingerMovements",
    "HandMovementDirection",
    "MindReading",
    "MotorImagery",
    "FaceDetection",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]

valid_uni_classifiers =[
    "Arsenal", "BOSS","Catch22","cBOSS","CIF","DrCIF","HC1","HC2","InceptionTime",
    "ProximityForest","ResNet","RISE","ROCKET","S-BOSS","STC","STSF","TDE",
    "TS-CHIEF","TSF","WEASEL"
]
valid_multi_classifiers = [
    "CBOSS", "CIF", "DTW_A", "DTW_D", "DTW_I", "gRSF", "InceptionTime","mrseql",
    "MUSE","ResNet","RISE","ROCKET","STC","TapNet","TSF"
]

def get_single_classifier_results_from_web(classifier, type="Univariate"):
    """Load the results for a single classifier on a single resample.

     Load from results into a dictionary of {problem_names: accuracy (numpy array)}.

     classifier: one of X
     type: string, either "Univariate" or "Multivariate"
    """
    if type == "Univariate":
        if not classifier in valid_uni_classifiers:
            raise Exception("Error, classifier ", classifier, "not in univariate set")
    elif type == "Multivariate":
        if not classifier in valid_multi_classifiers:
            raise Exception("Error, classifier ", classifier, "not in multivariate set")
    else:
        raise Exception("Type must be Univariate or Multivariate, you set it to ",type)

    url = "https://timeseriesclassification.com/results/ResultsByClassifier/"+type\
          +"/"+ classifier

    url = url+"_TESTFOLDS.csv"
    import requests
    response = requests.get(url)
    data = response.text
    split = data.split('\n')
    results = {}
    for i, line in enumerate(split):
        if len(line) > 0 and i > 0:
            all = line.split(",")
            res = np.array(all[1:]).astype(float)
            results[all[0]] = res
#    for inst in results:
#        print(inst, "  ", results[inst])
    return results

def get_averaged_results(datasets, classifiers, start=0, end=1, type="Multivariate"):
    """Extracts all results for UCR/UEA datasets on tsc.com for classifiers,
    then formats them into an array size n_datasets x n_classifiers.
    """
    if end<start:
        raise Exception("End resample smaller than start resample")
    results = np.zeros(shape=(len(datasets),len(classifiers)))
    cls_index = 0
    for cls in classifiers:
        selected = {}
        # Get all the results
        full_results = get_single_classifier_results_from_web(cls, type=type)
        # Extract the required ones
        data_index = 0
        for d in datasets:
            results[data_index][cls_index] = np.NaN
            if d in full_results:
                all_resamples = full_results[d]
                if len(all_resamples) >= end: # Average here
                    mean = all_resamples[start]
                    for i in range(start+1,end):
                        mean = mean+all_resamples[i]
                    results[data_index][cls_index] =mean/(end-start)
            data_index = data_index + 1
        cls_index = cls_index + 1
#    results = results.transpose()
    return results


res = get_single_classifier_results_from_web("ROCKET")

eeg_res = get_averaged_results(eeg_problems, ["ROCKET", "CIF", "InceptionTime",
                                              "HIVE-COTE"], end=30)
print(eeg_res[1])


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing
    """
    print(" Local Run")
    results_dir = "C:/Temp/EEG/"
    problem_path = "C://Data//EEG//"
    results_path = "C://Temp//"
    classifier = FreshPRINCE()
    dataset = "SelfRegulationSCP1"
#    load_and_run_classification_experiment(
#        problem_path=data_dir,
#        results_path=results_dir,
#        classifier=set_classifier(classifier, resample, tf),
#        cls_name="FreshPRINCE",
#        dataset=dataset,
#    )



