#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
wvemotion.physiology.ecg
--------------

"""
import numpy as np
import pandas as pd
from biosppy.signals import ecg
from hrv.utils import open_rri
from hrv.filters import moving_average
from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear
from errors import SourceTypeError, SourceDimensionError


class RInfo(object):
    """Information of R.

    Attributes:
        peaks: list of float, R-peak location indices.
        interval: list of float, R-peaks interval.
        hrv_time: dict, Heart rate variability of time domain anlysis.
        hrv_frequency: dict, Heart rate variability of frequency domain anlysis.
        hrv_nonlinear: dict, Heart rate variability of non-linear analysis.
    """
    pass


class ECG(object):
    """Process ECG signal.

    Attributes:
        ecg: narray of float, The original ecg signal data.
        rate: float, ECG frequency (Hz).
    """

    def __init__(self, constructor_source=None, signal_rate=1000.0):
        try:
            if isinstance(constructor_source, str):
                np_load = np.loadtxt(constructor_source)
                if np_load.ndim != 1:
                    raise SourceDimensionError(
                        'Just accept one column of ECG data.')
                self.__ecg = np_load
            elif isinstance(constructor_source, np.array):
                if constructor_source.ndim != 1:
                    raise SourceDimensionError(
                        'Just accept one dimension of ECG data.')
                self.__ecg = constructor_source
            elif isinstance(constructor_source, list):
                np_array = np.array(constructor_source, dtype=float)
                if np_array.ndim != 1:
                    raise SourceDimensionError(
                        'Just accept one dimension of ECG data.')
                self.__ecg = np_array
            else:
                raise SourceTypeError(
                    'Need invalid data file path or list or narray to new ECG object.')
        except SourceTypeError as error:
            raise error
        except ValueError as error:
            raise error
        except FileNotFoundError as error:
            raise error
        else:
            # delete the row include of nan
            self.__ecg = np.delete(
                self.__ecg, np.where(np.isnan(self.__ecg))[0], 0)
            self.__signal_rate = float(signal_rate)

    def get_ecg_narray(self):
        """Get narray of float of original ECG data.
        """
        return self.__ecg.copy() # deep copy

    def get_ecg_pandas(self, pd_type='Series'):
        """Get pandas Series or DataFrame of float of original ECG data.
        """
        pd_object = None
        if pd_type == 'Series':
            pd_object = pd.Series(self.__ecg)
        elif pd_type == 'DataFrame':
            pd_object = pd.DataFrame(self.__ecg)
        return pd_object

    def get_ecg_list(self):
        """Get list of float of original ECG data.
        """
        return np.ndarray.tolist(self.__ecg)

    def ecg_to_txt(self, des_path, format_str='%f', delimiter=' '):
        """Save ecg to txt file.
        """
        np.savetxt(des_path, self.__ecg, format_str, delimiter)

    def get_signal_rate(self):
        """Get this ECG source frequency (Hz).
        """
        return self.__signal_rate

    def rinfo_christov(self):
        """Get ECG R peaks by christov algorithm.
        """
        return ecg.christov_segmenter(self.__ecg, self.__signal_rate)

    def rinfo_engzee(self, threshold=0.48):
        """Get ECG R peaks by engzee algorithm.
        """
        return ecg.engzee_segmenter(self.__ecg, self.__signal_rate, threshold)

    def rinfo_gamboa(self, tolerance=0.002):
        """Get ECG R peaks by gamboa algorithm.
        """
        return ecg.gamboa_segmenter(self.__ecg, self.__signal_rate, tolerance)

    def rinfo_hamilton(self):
        """Get ECG R peaks by hamilton algorithm.
        """
        return ecg.hamilton_segmenter(self.__ecg, self.__signal_rate)

    def rinfo_ssf(self, threshold=20, before=0.03, after=0.01):
        """Get ECG R peaks by ssf algorithm.
        """
        return ecg.ssf_segmenter(self.__ecg, self.__signal_rate, threshold, before, after)

    def rinfo_correct(self, rpeaks=None, tolerance=0.05):
        """Get corrected ECG R peaks.
        """
        return ecg.correct_rpeaks(self.__ecg, rpeaks, self.__signal_rate, tolerance)

    def __calculate_rrinterval(self, rpeaks):
        """Calculate RR-Interval.
        """
        pass


# if __name__ == '__main__':
#     TEST = np.loadtxt('./data/test/extracted/Iris_kangxi_peyin_multiple.txt')
    # TEST = np.loadtxt('./test.txt')
    # print(TEST)
    # print(TEST.ndim)
    # print(TEST.shape)
    # TEST = np.where(np.isnan(TEST), 0, TEST)
    # TEST = np.delete(TEST, np.where(np.isnan(TEST))[0], 0)
    # TEST = np.delete(TEST, np.where(np.isnan(TEST)))
    # print(TEST.T)
    # TEST = [['1.1', '1.2', '1.3'], ['2.1', '2.2', '2.3'],
    #         ['3.1', '3.2', 'NaN'], ['4.1', 'NaN', '4.3']]
    # NP_TEST = np.array(TEST, dtype=float)
    # print(NP_TEST)
    # print(NP_TEST.ndim)
    # print(np.isnan(NP_TEST))
    # print(np.where(np.isnan(NP_TEST)))
    # NP_TEST = np.delete(NP_TEST, np.where(np.isnan(NP_TEST))[0], 0)
    # print(NP_TEST)
    # print(NP_TEST.T)
    # TEST = [1.1, 1.2, 1.3]
    # NP_TEST = np.array(TEST)
    # # PD_TEST = pd.DataFrame(NP_TEST)
    # print(len(NP_TEST))
