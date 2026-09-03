import numpy as np
from scipy.signal import find_peaks

def find_desaturations(signal, p=3):
    # First, get all local maxima
    peaks, props = find_peaks(signal, plateau_size=1)
    peaks = props['right_edges']
    events = []

    for i, peak in enumerate(peaks[:-1]):
        end = peaks[i+1]
        #low_point = peak + _argmin_last(signal[peak:end+1])
        # The paper does not state whether the first, middle or last minimum value is to be chosen.
        low_point = peak + np.argmin(signal[peak:end+1])
        drop = signal[peak] - signal[low_point]
        if drop >= p:
            events.append((peak, low_point, drop))

    return np.array(events)