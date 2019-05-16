#!/user/bin/env python3
# coding=utf-8

import numpy as np
from biosppy.signals import ecg
from biosppy.signals import eda
from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear
from hrv.utils import open_rri
from hrv.filters import moving_median

# np.set_printoptions(threshold=np.nan)

# load raw ECG signal
signal = np.loadtxt('./data/test/extracted/Iris_kangxi_peyin_ECG.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=1000.0, show=True)
print('R Peaks: ')
print(out['rpeaks'])
print('Heart Rate (BPM): ')
print(out['heart_rate'])# Instantaneous heart rate(bpm).
r_peaks = out['rpeaks']

# calculate RR-Interval
rr_interval = []
fs = open('./data/test/extracted/Iris_kangxi_peyin_RRI.txt',
          'w', encoding='utf-8')
for i in range(1, len(r_peaks)):
    interval = str(r_peaks[i] - r_peaks[i - 1]) + '\n'
    fs.write(interval)
    rr_interval.append(interval)

# RR-Interval to HRV
# filter
rri = open_rri('./data/test/extracted/Iris_kangxi_peyin_RRI.txt')
filt_rri = moving_median(rri, order=3)
# time domain
results = time_domain(filt_rri)
print(results)
# frequency domain
results = frequency_domain(
    rri=filt_rri,
    fs=4.0,
    method='welch',
    interp_method='cubic',
    detrend='linear'
)
print(results)
# non-linear domain
results = non_linear(filt_rri)
print(results)

# EDA
# load raw EDA signal
signal = np.loadtxt('./data/test/extracted/Iris_kangxi_peyin_EDA.txt')
# process it and plot
out = eda.eda(signal=signal, sampling_rate=1000.0,
              show=True, min_amplitude=0.1)
print('SCR Onsets: ')
print(out['onsets']) # Indices of SCR pulse onsets.
print('SCR Peaks: ')
print(out['peaks'])  # Indices of the SCR peaks.
print('SCR Amplitudes: ')
print(out['amplitudes']) # SCR pulse amplitudes.
