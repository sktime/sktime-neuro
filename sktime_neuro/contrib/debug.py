""" General working area for code development"""


import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mne_bids import BIDSPath, read_raw_bids


def bandwidth_examples():
    """Example of finding bandpowers from https://raphaelvallat.com/bandpower.html"""
    data = np.loadtxt('C:\Temp\data.txt')
    print("length of data =",len(data))
    sf = 100
    time = np.arange(data.size) / sf

    # Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(time, data, lw=1.5, color='k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    plt.xlim([time.min(), time.max()])
    plt.title('N3 sleep EEG data (F3)')
    plt.show()
    from scipy import signal
    # Define window length (4 seconds)
    win = 4 * sf
    freqs, psd = signal.welch(data, sf, nperseg=win)

    # Plot the power spectrum
#    sns.set(font_scale=1.2, style='white')
#    plt.figure(figsize=(8, 4))
    plt.plot(freqs, psd, color='k', lw=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    plt.xlim([0, freqs.max()])
    # Define delta lower and upper limits
    low, high = 0.5, 4

    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)

    # Plot the power spectral density and fill the delta area
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, psd, lw=2, color='k')
    plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (uV^2 / Hz)')
    plt.xlim([0, 10])
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    #plt.show()


if __name__ == "__main__":
    """Hacking around EEG data."""
    bandwidth_examples()
