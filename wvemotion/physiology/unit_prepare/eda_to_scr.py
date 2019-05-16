#!/user/bin/env python3
# coding=utf-8
import numpy as np
from biosppy.signals import eda

# load raw ECG signal
signal = np.loadtxt('./data/test/extracted/Iris_kangxi_peyin_EDA.txt')

# process it and plot
out = eda.eda(signal=signal, sampling_rate=1000.0, show=True, min_amplitude=0.1)
