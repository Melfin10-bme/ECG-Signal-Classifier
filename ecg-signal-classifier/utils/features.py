import numpy as np

def detect_r_peaks(ecg, fs=250, threshold=0.6):
    # Simple R-peak detection (absolute value > threshold * max)
    peaks = np.where(ecg > threshold * np.max(ecg))[0]
    return peaks

def extract_time_domain_features(ecg, fs=250):
    peaks = detect_r_peaks(ecg, fs)
    rr_intervals = np.diff(peaks) / fs

    if len(rr_intervals) < 2:
        return {"mean_hr": 0, "sdnn": 0, "rmssd": 0, "pnn50": 0}

    mean_hr = 60 / np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
    pnn50 = 100 * nn50 / len(rr_intervals)

    return {
        "mean_hr": float(mean_hr),
        "sdnn": float(sdnn),
        "rmssd": float(rmssd),
        "pnn50": float(pnn50),
    }

def classify_ecg(features):
    mean_hr = features["mean_hr"]
    sdnn = features["sdnn"]

    if mean_hr < 100 and sdnn > 0.03:
        return "Normal", 90.0
    else:
        return "Arrhythmia Suspected", 85.0
