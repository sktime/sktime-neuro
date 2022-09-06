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
    load_and_run_classification_experiment(
        problem_path=data_dir,
        results_path=results_dir,
        classifier=set_classifier(classifier, resample, tf),
        cls_name="FreshPRINCE",
        dataset=dataset,
    )



