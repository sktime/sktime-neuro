"""
Lots of datasets are in matlab .MAT files.
Thankfully we can use scipy to load these.
Unfortunatley, since these are just data-dumps from matlab sessions, we can't provide
an automated solution to parse these out into TS files.

As such, this file contains a set of utility functions that aim to make the pre-processing
steps less painful and faster.
"""

import scipy.io
import pandas as pd
import numpy as np


def load_matlab_mat(mat_file: str) -> dict:
    """
    A simple wrapper that laods a matlab .mat file and converts them to a pandas
    dataframe.
    """
    return scipy.io.loadmat(mat_file)


def extract_known_variable(mat: dict, key: str) -> np.ndarray:
    """
    For use when a known matlab vairable needs to be extracted from a .mat file.
    """
    return mat[key]


def mat_to_dataframe(mat) -> pd.DataFrame:
    """
    This assumes that all columns are 1 dimensional. Please pre-process before usage.
    """
    return pd.DataFrame.from_dict(mat)


def drop_ans(mat) -> dict:
    """
    Lots of matlab datasets are not cleaned and still contain the temporary variable
    ans that stores the answer of whatever calculation was previously performed in the REPL
    this generally serves no purpose, and breaks things (like mat_to_Dataframe)
    """
    try:
        popRes = mat.pop("ans", None)
        print(f"Popped \"ans\" value off of dict. It's value was {popRes}")
        return mat
    except KeyError:
        print("No ans value in dict, ignoring")
        return mat


def drop_metadata(mat) -> dict:
    """
    Drops matlab metadata such as the matlab header, and version information from
    the dict. These can cuase problems with the conversion process later.
    """
    try:
        popRes = mat.pop("__header__", None)
        popRes = mat.pop("__version__", None)
        return mat
    except KeyError:
        print("No metadata value in dict, ignoring")
        return mat


def extract_and_drop_globals(mat) -> (list, dict):
    """
    Matlab globals are often expoerted. Sometimes these are usefull but are not in a usable format out of the box.
    As such we remove them from the main matlab dict, and return them to the user so they can process them before
    later steps.
    """
    try:
        matglobals = mat.pop("__globals__", None)
        return (matglobals, mat)
    except KeyError:
        print("No globals found, ignoring")
        return mat


if __name__ == '__main__':
    m = load_matlab_mat("/CLA-SubjectJ-170508-3St-LRHand-Inter.mat")
    x = extract_known_variable(m, "x")
    m = drop_ans(m)
    m = drop_metadata(m)
    (g, m) = extract_and_drop_globals(m)
    df = mat_to_dataframe(m)
    print("Wait")
