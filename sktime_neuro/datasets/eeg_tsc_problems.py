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


def get_single_classifier_results_from_web(classifier):
    """Load the results for a single classifier on a single resample.

     Load from results into a dictionary of {problem_names: accuracy (numpy array)
    """
    url = "https://timeseriesclassification.com/results/ResultsByClassifier" \
          "/Multivariate/" \
          ""+classifier

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

def get_averaged_results(datasets, classifiers, resample=0):
    """Extracts all results for multivariate UEA datasets on tsc.com for classifiers,
    then formats them into an array size n_datasets x n_classifiers
    """
    results = np.zeros(shape=(len(datasets),len(classifiers)))
    cls_index = 0
    for cls in classifiers:
        selected = {}
        # Get all the results
        full_results = get_single_classifier_results_from_web(cls)
        # Extract the required ones
        data_index = 0
        for d in datasets:
            results[data_index][cls_index] = np.NaN
            if d in full_results:
                all_resamples = full_results[d]
                if(len(all_resamples)>resample): # Average here
                    results[data_index][cls_index] = all_resamples[resample]
            data_index = data_index + 1
        cls_index = cls_index + 1
#    results = results.transpose()
    return results


res = get_single_classifier_results_from_web("ROCKET")

eeg_res = get_averaged_results(eeg_problems, ["ROCKET", "CIF", "InceptionTime",
                                              "HIVE-COTE"])
print(eeg_res)


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



