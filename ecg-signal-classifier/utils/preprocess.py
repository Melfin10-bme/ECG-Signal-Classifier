import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, low=0.5, high=40):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(1, [low, high], btype="band")
    return filtfilt(b, a, signal)

def preprocess_ecg(ecg, fs=250):
    # Remove NaNs
    ecg = np.nan_to_num(ecg)
    # Bandpass filter
    ecg = bandpass_filter(ecg, fs)
    # Normalize
    ecg = (ecg - np.mean(ecg)) / np.std(ecg)
    return ecg
