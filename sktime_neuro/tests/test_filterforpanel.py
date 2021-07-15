# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sktime_neuro.transformations.panel.filterforpanel import FilterforPanel
from mne import filter


# Check that exception is raised for bad input arguments
@pytest.mark.parametrize("l_freq", [15, "str", -10, None, 1.5])
@pytest.mark.parametrize("h_freq", [10, "str", -10, None, 2.5])
def test_bad_input_args(l_freq, h_freq):
    np.random.seed(42)
    X = 0.02 * np.random.randn(100, 30, 1000)
    if not (
        isinstance(l_freq, (type(None), int, float))
        & isinstance(h_freq, (type(None), int, float))
    ):
        with pytest.raises((TypeError)):
            FilterforPanel(l_freq=l_freq, h_freq=h_freq, sfreq=250)

    elif (l_freq is not None) & (h_freq is not None):
        if not ((l_freq > 0) & (h_freq > 0)):
            with pytest.raises(ValueError):
                Filter = FilterforPanel(l_freq=l_freq, h_freq=h_freq, sfreq=250)
                Filter.fit_transform(X)
        if l_freq > h_freq:
            with pytest.raises(ValueError):
                Filter = FilterforPanel(l_freq=l_freq, h_freq=h_freq, sfreq=250)
                Filter.fit_transform(X)


# Test that keywords can be passed as arguments
@pytest.mark.parametrize(
    "kwargs",
    [
        {"picks": None},
        {"filter_length": "auto"},
        {"l_trans_bandwidth": "auto"},
        {"h_trans_bandwidth": "auto"},
        {"n_jobs": 1},
        {"method": "fir"},
        {"iir_params": None},
        {"copy": True},
        {"phase": "zero"},
        {"fir_window": "hamming"},
        {"fir_design": "firwin"},
        {"pad": "reflect_limited"},
        {"verbose": None},
    ],
)
def test_filter_kwargs(kwargs):
    np.random.seed(42)
    X = 0.02 * np.random.randn(100, 30, 1000)
    filterforpanel = FilterforPanel(sfreq=250, l_freq=1, h_freq=14, **kwargs)
    filterforpanel.fit_transform(X)


# Check that return values agree with those produced by mne
@pytest.mark.parametrize("sfreq", [100, 250])
@pytest.mark.parametrize("lfreq", [1, 10, None])
@pytest.mark.parametrize("hfreq", [11, 20, None])
def test_filter_results(sfreq, lfreq, hfreq):
    np.random.seed(42)
    X = 0.02 * np.random.randn(100, 30, 1000)

    # sklearn
    Filter = FilterforPanel(sfreq=sfreq, l_freq=lfreq, h_freq=hfreq)
    Xt1 = Filter.fit_transform(X)
    # mne
    Xt2 = filter.filter_data(X, sfreq=sfreq, l_freq=lfreq, h_freq=hfreq)

    assert np.allclose(Xt1, Xt2)
