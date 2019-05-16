#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
wvemotion.physiology.unit.ecg_to_rri
--------------
ECG to RR interval.
"""


import numpy as np
from biosppy.signals import ecg

# load raw ECG signal
signal = np.loadtxt('./data/test/extracted/Iris_kangxi_peyin_ECG.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=1000.0, show=True)
# ['ts', 'filtered', 'rpeaks', 'templates_ts', 'templates', 'heart_rate_ts', 'heart_rate']
# # http://biosppy.readthedocs.io/en/stable/biosppy.signals.html?highlight=ecg#biosppy-signals-ecg
print(out['rpeaks'])
# print(out.as_dict()['rpeaks'])

r_peaks = out['rpeaks']

rr_interval = []
fs = open('./data/test/extracted/Iris_kangxi_peyin_RRI.txt',
          'w', encoding='utf-8')
for i in range(1, len(r_peaks)):
    interval = str(r_peaks[i] - r_peaks[i-1]) + '\n'
    fs.write(interval)
    rr_interval.append(interval)
