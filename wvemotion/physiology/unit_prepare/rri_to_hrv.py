#!/user/bin/env python3
# coding=utf-8

from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear
from hrv.utils import open_rri
from hrv.filters import moving_median

import numpy as np

# def _moving_function(rri, order, func):
# 	offset = int(order / 2)

# 	filt_rri = np.array(rri.copy(), dtype=np.float64)
# 	for i in range(offset, len(rri) - offset, 1):
# 		filt_rri[i] = func(rri[i-offset:i+offset+1])

# 	return filt_rri

# def moving_median(rri, order=3):
# 	return _moving_function(rri, order, np.median)

rri = open_rri('./data/test/extracted/Iris_kangxi_peyin_RRI.txt')
filt_rri = moving_median(rri, order=3)
results = time_domain(filt_rri)
print(results)

results = frequency_domain(
    rri=filt_rri,
    fs=4.0,
    method='welch',
    interp_method='cubic',
    detrend='linear'
)
print(results)

results = non_linear(filt_rri)
print(results)
