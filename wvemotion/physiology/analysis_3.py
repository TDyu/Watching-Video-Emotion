#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze phsiology data.

Analyze phsiology data with BioSppy and hrv lib. and do visualization.
"""
import os

from biosppy.signals import ecg
from biosppy.signals import eda
from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

from loading import read_log_time_dict
from loading import get_data_narray_dict
from tools import normalize


ECG_NEED_LIST = ['HR_mean', 'HR_median', 'HR_variance', 'HR_sd', 'HRV_lf', 'HRV_lfnu', 'HRV_hf', 'HRV_hfnu', 'HRV_lf_hf', 'HRV_total_power', 'HRV_vlf']
EDA_NEED_LIST = ['EDA_ptp', 'EDA_diff', 'SCL', 'SCR_times']
# EDA_NEED_LIST = ['EDA_ptp', 'EDA_diff', 'SCL', 'SCL_normalized', 'SCR_times']

STAGE_LIST = ['base', '4-F', '5-SU', '21-D', '22-A', '28-SA', '35-H']


def ecg_extract(signal, sampling_rate=1000.0):
    """
    Args:
        signal: narray, ECG data.
        sampling_rate: float, Sampling rate of ECG data.

    Returns:
        A dictionary representation of the information of ECG.

    Notes:
        *Authors*

        - the bioSSPy dev team (https://github.com/PIA-Group/BioSPPy)

        *Dependencies*

        - biosppy

        *See Also*

        - BioSPPY: https://github.com/PIA-Group/BioSPPy
    """
    output = ecg.ecg(signal=signal, sampling_rate=1000.0, show=False)
    rr_interval = np.diff(output['rpeaks'])
    hrv_info = get_hrv(rr_interval)
    ecg_info = {'signal_time': output['ts'],
                'filtered': output['filtered'],
                'rpeaks': output['rpeaks'],
                'rr_interval': rr_interval,
                'heart_rate': output['heart_rate'],
                'heart_rate_ts': output['heart_rate_ts'],
                'hrv_time': hrv_info['time'],
                'hrv_frequency': hrv_info['frequency'],
                'hrv_non-linear': hrv_info['non-linear'],
                'HR_mean': np.mean(output['heart_rate']),
                 'HR_median': np.median(output['heart_rate']),
                 'HR_variance': np.var(output['heart_rate']),
                 'HR_sd': np.std(output['heart_rate']),
                 'HR_ptp': np.ptp(output['heart_rate']),
                 'HRV_lf': hrv_info['frequency']['lf'],
                 'HRV_lfnu': hrv_info['frequency']['lfnu'],
                 'HRV_hf': hrv_info['frequency']['hf'],
                 'HRV_hfnu': hrv_info['frequency']['hfnu'],
                 'HRV_lf_hf': hrv_info['frequency']['lf_hf'],
                 'HRV_total_power': hrv_info['frequency']['total_power'],
                 'HRV_vlf': hrv_info['frequency']['vlf']}
    return ecg_info


def get_hrv(rr_interval):
    """Get three domain heart rate variability.

    Get time domain, frequency domain, and non-linear domain heart rate variability.

    Args:
        rr_interval: narray, RR-interval.

    Returns:
        A dictionary representation of the three domain HRV.

    Notes:
        *Authors*

        - the hrv dev team (https://github.com/rhenanbartels/hrv)

        *Dependencies*

        - hrv
        - numpy

        *See Also*

        - hrv: https://github.com/rhenanbartels/hrv
    """
    if np.median(rr_interval) < 1:
        rr_interval *= 1000
    time_domain_analysis = time_domain(rr_interval)
    frequency_domain_analysis = frequency_domain(
        rri=rr_interval,
        fs=4.0,
        method='welch',
        interp_method='cubic',
        detrend='linear'
    )
    non_linear_domain_analysis = non_linear(rr_interval)
    hrv_info = {'time': time_domain_analysis,
                'frequency': frequency_domain_analysis,
                'non-linear': non_linear_domain_analysis}
    return hrv_info


def eda_extract(signal, sampling_rate=1000.0, min_amplitude=0.1):
    """
    Args:
        signal: narray, EDA data.
        sampling_rate: float, Sampling rate of EDA data.
        min_amplitude: float, Minimum treshold by which to exclude SCRs.

    Returns:
        A dictionary representation of the information of EDA.

    Notes:
        *Authors*

        - the bioSSPy dev team (https://github.com/PIA-Group/BioSPPy)

        *Dependencies*

        - biosppy

        *See Also*
        - BioSPPY: https://github.com/PIA-Group/BioSPPy
    """
    output = eda.eda(signal=signal, sampling_rate=1000.0,
                     show=False, min_amplitude=0.1)
    eda_info = {'signal_time': output['ts'],
                'filtered': output['filtered'],
                'onsets': output['onsets'],
                'peaks': output['peaks'],
                'amplitudes': output['amplitudes'],
                'EDA_ptp': np.ptp(signal),
                'EDA_diff': signal[-1] - signal[0],
                'SCL': np.mean(signal),
                # 'SCL_normalized': np.mean(normalize(signal)),
                'SCR_times': output['onsets'].size}
    return eda_info


class PersonInfo(object):
    """包裝每個人的每一段的數值
    """
    def __init__(self, time_name, stage_info_dict, value_type='original'):
        super(PersonInfo, self).__init__()
        self.stage_info_dict = stage_info_dict
        # print(self.stage_info_dict)
        if value_type == 'min-max':
            self.__min_max_normalization()
        elif value_type == 'z-score':
            self.__z_score()

    def __min_max_normalization(self):
        # https://www.biaodianfu.com/python-normalization-method.html
        need_list = ECG_NEED_LIST + EDA_NEED_LIST
        for need in need_list:
            key_name_need_list = []
            key_name_corresponding_list = []
            for stage in STAGE_LIST:
                    key_name = stage + '_' + need
                    key_name_corresponding_list.append(key_name)
                    key_name_need_list.append(self.stage_info_dict[key_name])
            key_name_need_list = np.array(key_name_need_list)
            min_value = np.min(key_name_need_list)
            max_value = np.max(key_name_need_list)
            index = 0
            for key_name in key_name_corresponding_list:
                self.stage_info_dict[key_name] = (
                    key_name_need_list[index] - min_value) / (max_value - min_value)
                index += 1
            # min_max_scaler = preprocessing.MinMaxScaler()
            # x_train_minmax = min_max_scaler.fit_transform(np.array(key_name_need_list))
            # index = 0
            # for key_name in key_name_corresponding_list:
            #     self.stage_info_dict[key_name] = x_train_minmax[index]
            #     index += 1

    def __z_score(self):
        # https://www.cnblogs.com/chaosimple/p/4153167.html
        need_list = ECG_NEED_LIST + EDA_NEED_LIST
        for need in need_list:
            key_name_need_list = []
            key_name_corresponding_list = []
            for stage in STAGE_LIST:
                    key_name = stage + '_' + need
                    key_name_corresponding_list.append(key_name)
                    key_name_need_list.append(self.stage_info_dict[key_name])
            key_name_need_list = np.array(key_name_need_list)
            mean_value = np.mean(key_name_need_list)
            std_value = np.std(key_name_need_list, ddof=1)
            index = 0
            for key_name in key_name_corresponding_list:
                self.stage_info_dict[key_name] = (
                    key_name_need_list[index] - mean_value) / std_value
                index += 1
            # z_score_array = preprocessing.scale(np.array(key_name_need_list))
            # index = 0
            # for key_name in key_name_corresponding_list:
            #     self.stage_info_dict[key_name] = z_score_array[index]
            #     index += 1


def culculate_info(time_name, corresponding_data_dict, value_type='original'):
    """轉化對應時間段的量測數據成為有意義的生理數值
    """
    stage_info_dict = {}
    for stage_name, psy_data in corresponding_data_dict.items():
        if stage_name != 'measuring':
            ecg_info = ecg_extract(psy_data.ecg)
            eda_info = eda_extract(psy_data.eda)
            for need in ECG_NEED_LIST:
                key_name = stage_name + '_' + need
                stage_info_dict[key_name] = ecg_info[need]
            for need in EDA_NEED_LIST:
                key_name = stage_name + '_' + need
                stage_info_dict[key_name] = eda_info[need]
    return PersonInfo(time_name, stage_info_dict, value_type=value_type)


def deal_all_people_data(log_parent_path, source_parent_path, key_list, value_type='original', get_type='original'):
    """會循環所指定的資料夾（所以資料夾要設定好），處理所有人的資料
    """
    people_info_list = []
    files = os.listdir(source_parent_path)
    for f in files:
        relative_path = os.path.join(source_parent_path, f)
        if os.path.isfile(relative_path):
            # Deal path.
            name_index = f.find('_')
            file_exten_index = f.find('.txt')
            time_name = f[:name_index]
            all_name = f[:file_exten_index]
            log_path = log_parent_path
            if log_parent_path[-1] != '/':
                log_path += '/' + all_name + 'experiment.log'
            else:
                log_path += all_name + 'experiment.log'
            source_path = relative_path
            # Deal time. (Need to deal time and then deal data.)
            key_time_dict = read_log_time_dict(log_path, key_list)
            # Deal data.
            corresponding_data_dict = get_data_narray_dict(
                key_time_dict, source_path, get_type=get_type)
            # Package.
            # * 先取得所有人的data會再那一關就MemoryError...但是取一個人算一個就不會...
            people_info_list.append(culculate_info(
                time_name, corresponding_data_dict, value_type=value_type))
    return people_info_list


def draw_boxplot(people_info_list, output_path, value_type='original'):
    """畫出每一種生理意義數值的boxplot
    """
    need_list = ECG_NEED_LIST + EDA_NEED_LIST
    for need in need_list:
        key_name_list = []
        for stage in STAGE_LIST:
            key_name = stage + '_' + need
            info_dict = {}
            for people_info in people_info_list:
                if key_name not in info_dict:
                    info_dict[key_name] = [people_info.stage_info_dict[key_name]]
                else:
                    info_dict[key_name].append(people_info.stage_info_dict[key_name])
            key_name_list.append(np.array(info_dict[key_name]))
        need_array = np.array(key_name_list).T
        # Plot.
        df = pd.DataFrame(need_array, columns=STAGE_LIST)
        df.boxplot()
        plt.title(need)
        current_output = output_path
        if output_path[-1] != '/':
            current_output += '/' + need + '_' + value_type + '.png'
        else:
            current_output += need + '_' + value_type + '.png'
        plt.savefig(current_output)
        plt.cla()
        # plt.show()
        
    # for people_info in people_info_list:
    #     for stage_info_key, value in people_info.stage_info_dict.items():
    #         if stage_info_key not in info_dict:
    #             info_dict[stage_info_key] = [value]
    #         else:
    #             info_dict[stage_info_key].append(value)


if __name__ == '__main__':
    key_list = ['4-F', '5-SU', '21-D', '22-A', '28-SA', '35-H']
    log_parent_path = './data/lab/log/'
    source_parent_path = './data/lab/source/'
    output_path = './data/lab/chart_output/'
    # 取出原始量測數據時要不要標準化
    get_type = 'original'
    # get_type = 'z-score'
    # get_type = 'min-max'
    # value_type = 'z-score'
    # 分段時再行標準化
    value_type = 'min-max'
    # value_type = 'original'
    # people_info_list = deal_all_people_data(
    #     log_parent_path, source_parent_path, key_list, value_type=value_type)
    # draw_boxplot(people_info_list, output_path, value_type=value_type)
    people_info_list = deal_all_people_data(
        log_parent_path, source_parent_path, key_list, value_type=value_type, get_type=get_type)
    draw_boxplot(people_info_list, output_path, value_type=value_type)
