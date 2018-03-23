#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze phsiology data.

Analyze phsiology data with BioSppy and hrv lib. and do visualization.
"""
from extract import extract_multiple_type_narray
from biosppy.signals import ecg
from biosppy.signals import eda
from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear
from matplotlib import pyplot as plt
import numpy as np
import os


def analyze(source_path, inner_draw=False):
    """Extract labchart data and output visual analysis.

    Extract one person's ECG and EDA data to 2x1 numpy array by
    extract_multiple_type_narray. (First row is ECG, another row is EDA.)
    Notice that the 2nd column of original txt data need to be ECG data, and 3rd column need
    to be EDA data.
    Afterwards, use ecg_extract and eda_extract to extract the wanted information.
    At last, draw chart.
    """
    signals = extract_multiple_type_narray(source_path, 1, 2)
    ecg_info = ecg_extract(signals[0], inner_draw=inner_draw)
    eda_info = eda_extract(signals[1], inner_draw=inner_draw)
    return ecg_info, eda_info


def ecg_extract(signal, sampling_rate=1000.0, inner_draw=False):
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
    output = ecg.ecg(signal=signal, sampling_rate=1000.0, show=inner_draw)
    rr_interval = np.diff(output['rpeaks'])
    hrv_info, hrv_per_minute_info = get_hrv_all(
        output['rpeaks'], interval_second=0.001)
    hr_per_minute_info = _count_hr_per_minute_info(
        output['heart_rate_ts'], output['heart_rate'])
    ecg_info = {'signal_time': output['ts'],
                'filtered': output['filtered'],
                'rpeaks': output['rpeaks'],
                'rr_interval': rr_interval,
                'heart_rate': output['heart_rate'],
                'heart_rate_ts': output['heart_rate_ts'],
                'hr_per_minute_info': hr_per_minute_info,
                'hrv_time': hrv_info['time'],
                'hrv_frequency': hrv_info['frequency'],
                'hrv_non-linear': hrv_info['non-linear'],
                'hrv_per_minute_info': hrv_per_minute_info}
    return ecg_info


def _count_hr_per_minute_info(hr_ts, hr_list):
    hr_per_minute_info = {}
    hr_per_minute_list = _count_hr_per_minute(hr_ts, hr_list)
    hr_per_minute_info['mean'] = _count_mean_per_minute(hr_per_minute_list)
    hr_per_minute_info['median'] = _count_median_per_minute(
        hr_per_minute_list)
    hr_per_minute_info['variance'] = _count_variance_per_minute(
        hr_per_minute_list)
    hr_per_minute_info['sd'] = _count_sd_per_minute(hr_per_minute_list)
    hr_per_minute_info['max_diff'] = _count_max_diff_per_minute(
        hr_per_minute_list)
    return hr_per_minute_info


def _count_hr_per_minute(hr_ts, hr_list):
    """Cut heart rate into multiple narrays for each minute.

    Cut heart rate into multiple narrays for each minute. Furthermore, pack into
    a list (Index is minute).

    Args:
        hr_ts: narray, Heart rate time axis reference (seconds).
        hr_list: narray, Instantaneous heart rate (bpm).
    
    Returns:
        A list representation of heart rate values in each minute. Index is minute,
        and each item is a narray representation of the heart rate in this minute.
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    minute = 1
    hr_per_minute_list = [[]]
    for i in range(0, hr_list.size):
        if hr_ts[i] <= minute * 60.0:
            pass
        else:
            hr_per_minute_list[minute - 1] = np.array(
                hr_per_minute_list[minute - 1])
            hr_per_minute_list.append([])
            minute += 1
        hr_per_minute_list[minute - 1].append(hr_list[i])
    hr_per_minute_list[minute - 1] = np.array(
        hr_per_minute_list[minute - 1])
    return hr_per_minute_list


def _count_mean_per_minute(per_minute_list):
    """Calculate the mean per minute.

    Calculate the mean per minute. Furthermore, pack into a list
    (Index is minute).

    Args:
        per_minute_list: list of narray, Physiological values in each minute.
    
    Returns:
        A narray representation of mean (Index is minute).
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    mean_per_minute = []
    for minute_narray in per_minute_list:
        mean_per_minute.append(np.mean(minute_narray))
    return np.array(mean_per_minute)


def _count_median_per_minute(per_minute_list):
    """Calculate the median per minute.

    Calculate the median per minute. Furthermore, pack into a list
    (Index is minute).

    Args:
        per_minute_list: list of narray, Physiological values in each minute.
    
    Returns:
        A narray representation of median (Index is minute).
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    median_per_minute = []
    for minute_narray in per_minute_list:
        median_per_minute.append(np.median(minute_narray))
    return np.array(median_per_minute)


def _count_variance_per_minute(per_minute_list):
    """Calculate the variance per minute.

    Calculate the variance per minute. Furthermore, pack into a list
    (Index is minute).

    Args:
        per_minute_list: list of narray, Physiological values in each minute.
    
    Returns:
        A narray representation of variance (Index is minute).
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    variance_per_minute = []
    for minute_narray in per_minute_list:
        variance_per_minute.append(np.var(minute_narray))
    return np.array(variance_per_minute)


def _count_sd_per_minute(per_minute_list):
    """Calculate the standard deviation per minute.

    Calculate the standard deviation per minute. Furthermore, pack into a list
    (Index is minute).

    Args:
        per_minute_list: list of narray, Physiological values in each minute.
    
    Returns:
        A narray representation of standard deviation (Index is minute).
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    sd_per_minute = []
    for minute_narray in per_minute_list:
        sd_per_minute.append(np.std(minute_narray))
    return np.array(sd_per_minute)


def _count_max_diff_per_minute(per_minute_list):
    """Calculate the maximum difference per minute.

    Calculate the maximum difference per minute. Furthermore, pack into a list
    (Index is minute).

    Args:
        per_minute_list: list of narray, Physiological values in each minute.
    
    Returns:
        A narray representation of maximum difference (Index is minute).
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    max_diff_per_minute = []
    for minute_narray in per_minute_list:
        max_diff_per_minute.append(np.ptp(minute_narray))
    return np.array(max_diff_per_minute)


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


def get_hrv_all(rpeaks, interval_second=0.001):
    """
    Notes:
        *Dependencies*
        
        - numpy
    """
    rrinterval_all = np.diff(rpeaks)
    hrv_all = get_hrv(rrinterval_all)
    hrv_per_minute_dict = _count_hrv_per_minute(rpeaks, interval_second)
    return hrv_all, hrv_per_minute_dict


def _count_hrv_per_minute(rpeaks, interval_second=0.001):
    rpeaks_per_minute_list = _count_rpeaks_per_minute(
        rpeaks, interval_second=0.001)
    rrinterval_per_minute_list = _count_rrinterval_per_minute(
        rpeaks_per_minute_list)
    hrv_per_minute = []
    hrv_per_minute_dict = {
        'time': {
            'mhr': [],
            'mrri': [],
            'nn50': [],
            'pnn50': [],
            'rmssd': [],
            'sdnn': []},
        'frequency': {
            'hf': [],
            'hfnu': [],
            'lf': [],
            'lf_hf': [],
            'lfnu': [],
            'total_power': [],
            'vlf': []},
        'non-linear': {
            'sd1': [],
            'sd2': []
        }}
    for rrinterval in rrinterval_per_minute_list:
        # UserWarning: nperseg = 256 is greater than input length  = xxx, using nperseg = xxx
        hrv_per_minute.append(get_hrv(rrinterval))
    for hvr_info in hrv_per_minute:
        hrv_per_minute_dict['time']['mhr'].append(
            hvr_info['time']['mhr'])
        hrv_per_minute_dict['time']['mrri'].append(
            hvr_info['time']['mrri'])
        hrv_per_minute_dict['time']['nn50'].append(
            hvr_info['time']['nn50'])
        hrv_per_minute_dict['time']['pnn50'].append(
            hvr_info['time']['pnn50'])
        hrv_per_minute_dict['time']['rmssd'].append(
            hvr_info['time']['rmssd'])
        hrv_per_minute_dict['time']['sdnn'].append(
            hvr_info['time']['sdnn'])
        hrv_per_minute_dict['frequency']['hf'].append(
            hvr_info['frequency']['hf'])
        hrv_per_minute_dict['frequency']['hfnu'].append(
            hvr_info['frequency']['hfnu'])
        hrv_per_minute_dict['frequency']['lf'].append(
            hvr_info['frequency']['lf'])
        hrv_per_minute_dict['frequency']['lf_hf'].append(
            hvr_info['frequency']['lf_hf'])
        hrv_per_minute_dict['frequency']['lfnu'].append(
            hvr_info['frequency']['lfnu'])
        hrv_per_minute_dict['frequency']['total_power'].append(
            hvr_info['frequency']['total_power'])
        hrv_per_minute_dict['frequency']['vlf'].append(
            hvr_info['frequency']['vlf'])
        hrv_per_minute_dict['non-linear']['sd1'].append(
            hvr_info['non-linear']['sd1'])
        hrv_per_minute_dict['non-linear']['sd2'].append(
            hvr_info['non-linear']['sd2'])
    return hrv_per_minute_dict


def _count_rpeaks_per_minute(rpeaks, interval_second=0.001):
    """Cut R-Peaks into multiple narrays happened at each minute.

    Cut appened at multiple narrays happened at each minute. Furthermore, pack
    into a list (Index is minute).

    Args:
        rpeaks, narray, R-peak location indices..
        interval_second: float, Interval second (1/ Sampling rate).
    
    Returns:
        A list representation of R-peak in each minute. Index is minute, and
        each item is a narray representation of the R-Peaks in this minute.
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    minute = 1
    rpeaks_per_minute_list = [[]]
    for i in range(0, rpeaks.size):
        if rpeaks[i] * interval_second // 60 < minute:
            pass
        else:
            rpeaks_per_minute_list[minute -
                                   1] = np.array(rpeaks_per_minute_list[minute - 1])
            rpeaks_per_minute_list.append([])
            minute += 1
        rpeaks_per_minute_list[minute - 1].append(rpeaks[i])
    rpeaks_per_minute_list[minute -
                           1] = np.array(rpeaks_per_minute_list[minute - 1])
    return rpeaks_per_minute_list


def _count_rrinterval_per_minute(rpeaks_per_minute_list):
    """Calculate RR-interval per minute.

    Args:
        rpeaks_per_minute_list: list of narray, R-peaks in each minute.
    
    Returns:
        A list of narray, RR-interval per minute.
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    rrinterval_per_minute_list = []
    for rpeaks in rpeaks_per_minute_list:
        rrinterval_per_minute_list.append(np.diff(rpeaks))
    return rrinterval_per_minute_list


def eda_extract(signal, sampling_rate=1000.0, min_amplitude=0.1, inner_draw=False):
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
                     show=inner_draw, min_amplitude=0.1)
    eda_per_minute_list = _count_eda_per_minute(
        output['filtered'], interval_second=1 / sampling_rate)
    scl_per_minute_info = _get_scl_per_minute_info(eda_per_minute_list)
    scr_per_minute_info = _get_scr_per_minute_info(
        output['onsets'], output['amplitudes'])
    eda_info = {'signal_time': output['ts'],
                'filtered': output['filtered'],
                'onsets': output['onsets'],
                'peaks': output['peaks'],
                'amplitudes': output['amplitudes'],
                'scl_per_minute_info': scl_per_minute_info,
                'scr_per_minute_info': scr_per_minute_info}
    return eda_info


def _get_scl_per_minute_info(eda_per_minute_list):
    """Get statistical value of SCL.

    Args:
        eda_per_minute_list: list of narray, EDA signals in each minute.
    
    Returns:
        A dictionary representation of statistical of SCL.
    """
    scl_per_minute_info = {
        'scl': _count_scl_per_minute(eda_per_minute_list),
        'max_diff_per_minute': _count_max_diff_per_minute(eda_per_minute_list),
        'inout_diff_per_minute': _count_eda_inout_diff_per_minute(
            eda_per_minute_list)
    }
    return scl_per_minute_info


def _get_scr_per_minute_info(onsets, amplitudes):
    """Get statistical value of SCR.

    Args:
        onsets: narray, Indices of SCR pulse onsets.
        amplitudes: narray, SCR pulse amplitudes.
    
    Returns:
        A dictionary representation of statistical of SCR.
    """
    times_per_minute = _count_times_scr_per_minute(onsets)
    scr_per_minute_info = {
        'times_per_minute':  times_per_minute,
        'amplitudes_per_minute_info': _get_scr_amplitudes_info_per_minute(
            times_per_minute, amplitudes)
    }
    return scr_per_minute_info
    


def _count_scl_per_minute(eda_per_minute_list):
    """Calculate SCL.

    Args:
        eda_per_minute_list: list of narray, EDA signals in each minute.
    
    Returns:
        A list representation of SCL.
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    scl_per_minute = []
    for eda_per_minute in eda_per_minute_list:
        scl_per_minute.append(np.mean(eda_per_minute))
    return scl_per_minute


def _count_eda_per_minute(filtered_signal, interval_second=0.001):
    """Cut filtered EDA signals into multiple narrays for each minute.

    Args:
        filtered_signal: narray, Filtered EDA signal.
        interval_second: float, Interval second (1/ Sampling rate).
    
    Returns:
        A narray representation of EDA signals in each minute.
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    minute = 1
    eda_per_minute_list = [[]]
    for i in range(0, filtered_signal.size):
        if (i + 1) * interval_second // 60 < minute:
            pass
        else:
            eda_per_minute_list[minute -
                                   1] = np.array(eda_per_minute_list[minute - 1])
            eda_per_minute_list.append([])
            minute += 1
        eda_per_minute_list[minute - 1].append(filtered_signal[i])
    eda_per_minute_list[minute -
                           1] = np.array(eda_per_minute_list[minute - 1])
    return eda_per_minute_list


def _count_eda_inout_diff_per_minute(eda_per_minute_list):
    """Calculate the value when leaving is subtracted from the value when it comes in.

    Args:
        eda_per_minute_list: list of narray, EDA signals in each minute.
    
    Returns:
        A list representation of difference that value of leaving minus value of
        coming in each time.
    """
    eda_inout_diff_per_minute = []
    for eda_per_minute in eda_per_minute_list:
        eda_inout_diff_per_minute.append(
            eda_per_minute[-1] - eda_per_minute[1])
    return eda_inout_diff_per_minute


def _count_times_scr_per_minute(scr_info, interval_second=0.001):
    """Calculate SCR times per minute.

    Args:
        scr_info: narray, Indices of the SCR peaks or onsets.
        interval_second: float, Interval second (1/ Sampling rate).

    Returns:
        A list representation of times in each minute.
    """
    print(scr_info[-1])
    times_scr_per_minute = [0]
    minute = 1
    for info in scr_info:
        if info * interval_second // 60 < minute:
            pass
        else:
            times_scr_per_minute.append(0)
            minute += 1
        times_scr_per_minute[minute - 1] += 1
    print(len(times_scr_per_minute))
    print(times_scr_per_minute)
    return times_scr_per_minute


def _get_scr_amplitudes_info_per_minute(times_scr_per_minute, amplitudes):
    """Get multiple statistical values of SCR.

    Args:
         times_scr_per_minute: list, SCR times in each minute.
        amplitudes: narray, SCR pulse amplitudes.
    
    Returns:
        A dictionary representation of statistical information of SCR.
    """
    amplitudes_per_minute = _count_amplitudes_per_minute(
        times_scr_per_minute, amplitudes)
    scr_amplitudes_info = {
        'mean': _count_mean_per_minute(amplitudes_per_minute),
        'median': _count_median_per_minute(amplitudes_per_minute),
        'variance': _count_variance_per_minute(amplitudes_per_minute),
        'sd': _count_sd_per_minute(amplitudes_per_minute),
        'max_diff': _count_max_diff_per_minute(amplitudes_per_minute)}
    return scr_amplitudes_info


def _count_amplitudes_per_minute(times_scr_per_minute, amplitudes):
    """Cut SCR amplitudes into multiple narrays for each minute.

    Args:
        times_scr_per_minute: list, SCR times in each minute.
        amplitudes: narray, SCR pulse amplitudes.
    
    Returns:
        A list representation of amplitudes in each minute.
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    amplitudes_per_minute = []
    amplitudes_list = amplitudes.tolist()
    for i in range(0, len(times_scr_per_minute)):
        amplitudes_per_minute.append([])
        for index in [0] * times_scr_per_minute[i]:
            amplitudes_per_minute[i].append(amplitudes_list.pop(index))
        amplitudes_per_minute[i] = np.array(amplitudes_per_minute[i])
    return amplitudes_per_minute


def reload_label_file(data, data_reload):
    """Reload questionnaire emotion label file for encoding.
    """
    with open(data_reload, 'w', encoding='utf-8') as label_file2:
        with open(data, 'r', encoding='utf-8') as label_file:
            for line in label_file:
                if '\ufeff' in line:
                    line = line.replace('\ufeff', '')
                label_file2.write(line)


def get_corrected_quest_label(label_data, vedio_time_list,interval_time=(0, 0),
                              pre_spaces=0):
    """Get scores of emotion labels with corrected amount.

    Get the scores per minute of each emotion. And the amount of each score need
    to be able to match with info of heart rate per minute.

    Args:
        label_data: str, Path of questionnaire labels.
        vedio_time_list: list of tuple, Time tuples (minute, second) of vedios.
        interval_time: tuple, Interval time of main experiment videos.
        pre_spaces: int, The spaces number during preparation.
    
    Returns:
         A dictionary representation of label value of each emotion.

    Notes:
        *Dependencies*
        
        - numpy
    """
    labels = np.loadtxt(label_data, dtype=int)
    label_should_amount = _count_should_label(vedio_time_list, interval_time)
    return _correct_label_value(
        labels, vedio_time_list, label_should_amount, interval_time, pre_spaces)


def _count_should_label(vedio_time_list, interval_time=(0, 0)):
    """Calculate should have how many questionnaire labels of these vedios.

    Args:
        vedio_time_list: list of tuple, Time tuples (minute, second) of vedios.
        interval_time: tuple, Interval time of main experiment videos.
    
    Returns:
        A list representation that label should have amount of each main
        experiment video.
    """
    label_amount = []
    for time in vedio_time_list:
        if time != interval_time:
            if time[1] > 0:
                label_amount.append(time[0] + 1)
            else:
                label_amount.append(time[0])
    return label_amount


def _correct_label_value(labels, vedio_time_list, label_should_amount,
                         interval_time=(0, 0), pre_spaces=0):
    """Correct label value to be enough for matching with info of heart rate per minute.

    Args:
        labels: list of narray, Questionnaire labels of these vedios.
        vedio_time_list: list of tuple, Time tuples (minute, second) of vedios.
        label_should_amount: list, Label should have amount of each main
            experiment video.
        interval_time: tuple, Interval time of main experiment videos.
        pre_spaces: int, The spaces number during preparation.
    
    Returns:
        A dictionary representation of label value of each emotion.
    """
    need_makeup_index = _find_makeup_index(
        vedio_time_list, label_should_amount, interval_time)
    fixed_labels = _fix_label_value(labels, need_makeup_index, pre_spaces)
    corrected_label_dict = {
        'Angry': fixed_labels.T[0],
        'Disgusting': fixed_labels.T[1],
        'Fear': fixed_labels.T[2],
        'Happy': fixed_labels.T[3],
        'Sad': fixed_labels.T[4],
        'Surprise': fixed_labels.T[5]}
    return corrected_label_dict


def _find_makeup_index(vedio_time_list, label_should_amount, interval_time=(0, 0)):
    """Find the index of the points which need to make up.

    Args:
        vedio_time_list: list of tuple, Time tuples (minute, second) of vedios.
        label_should_amount: list, Label should have amount of each main
            experiment video.
        interval_time: tuple, Interval time of main experiment videos.
    
    Returns:
        A list representation of the index of the points which needed to make up.
    """
    need_makeup_index = []
    current_time = [vedio_time_list[0][0], vedio_time_list[0][1]]
    mackup_amount = 0
    interval_times = 0
    for i in range(1, len(vedio_time_list)):
        if i != len(vedio_time_list) - 1 and vedio_time_list[i] == interval_time:
            pre_time = current_time
            current_time = current_time = _find_next_time(
                current_time, vedio_time_list[i])
            if current_time[0] == pre_time[0]:

                # "i" is the index which "vedio" need to make up.
                # In order to obtain the correct accumulated ceiling index value
                # of label_should_amount, i need to minus 1 because that this i
                # is from 1, and then need to minus interval_times to pass index
                # the vedio which are interval videos. After accumulating the
                # number of labels that should have passed, need to add number
                # numbar generated due to compensation.

                need_makeup_index.append(
                    sum(label_should_amount[:i - 1 - interval_times]) + mackup_amount)
                mackup_amount += 1
        else:
            current_time = _find_next_time(current_time, vedio_time_list[i])
        if interval_time != (0, 0) and vedio_time_list[i] == interval_time and i != 1:
            interval_times += 1
    return need_makeup_index


def _find_next_time(pre_time, add_time):
    """Find the next minute:second time.

    Args:
        pre_time: tuple, Previous time (minute, second).
        add_time: tuple, Add time (minute, second).
    
    Returns:
        A tuple representation of current time (minute, second).
    """
    next_time = [pre_time[0] + add_time[0],
                    pre_time[1] + add_time[1]]
    if next_time[1] > 60:
        next_time[0] += 1
        next_time[1] -= 60
    return next_time


def _fix_label_value(labels, need_makeup_index, pre_spaces=0):
    """Make up the label value with 0.

    Args:
        labels: list of narray, Questionnaire labels of these vedios.
        need_makeup_index: list, The index of the points which needed to make up.
        pre_spaces: int, The spaces number during preparation.
    
    Returns:
        A list of narray representation of maken up label value that can use to
        match heart rate info per minute.
    
    Notes:
        *Dependencies*
        
        - numpy
    """
    fixed_labels = labels
    make_up = [0, 0, 0, 0, 0, 0]
    for index in need_makeup_index:
        fixed_labels = np.insert(fixed_labels, index, values=make_up, axis=0)
    if pre_spaces != 0:
        for index in [0] * pre_spaces:
            fixed_labels = np.insert(
                fixed_labels, index, values=make_up, axis=0)
    return fixed_labels


def draw_scratter(corrected_label_dict, per_minute_info, save_folder, phy_type):
    """Draw scratter chart with physiological information and questionnaire
    emotion labels.

    Draw scratter chart. X axis is one type physiological data. Y axis is one
    type questionnaire emotion labels.

    Args:
        corrected_label_dict: dict of narray, Corrected label value of each emotion.
        per_minute_info: dict of narray, Information per minute.
        sace_folder: str, Specified image storage path.
        phy_type: str, Type of physiological information (HR, HRV_F, HRV_T, HRV_N,...).
            HR => ecg_info['hr_per_minute_info']
            HRV_F => ecg_info['hrv_per_minute_info']['frequency']
            HRV_T => ecg_info['hrv_per_minute_info']['time']
            HRV_N => ecg_info['hrv_per_minute_info']['non-linear']
            SCL => eda_info['scl_per_minute_info']
            SCR_TIMES => {'SCR_TIMES': eda_info['scr_per_minute_info']['times_per_minute']}
            SCR_AMP => eda_info['scr_per_minute_info']['amplitudes_per_minute_info']

    Notes:
        *Dependencies*
        
        - matplotlib
    """
    COLOR = {
        'Angry': 'red',
        'Disgusting': 'pink',
        'Fear': 'black',
        'Happy': 'yellow',
        'Sad': 'blue',
        'Surprise': 'purple'}
    save_folder_path = save_folder + phy_type + '/'
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    for info in per_minute_info.keys():
        for emotion in corrected_label_dict.keys():
            img_name = save_folder_path + info + '_' + emotion + '.png'
            plt.xlabel(info + ' of ' + phy_type + '(minute)')
            plt.ylabel('scores of ' + emotion + ' (minute)')
            plt.scatter(per_minute_info[info], corrected_label_dict[emotion],
                        c=COLOR[emotion], s=25, alpha=1.0, marker='o')
            plt.savefig(img_name)
            plt.clf()


if __name__ == '__main__':
    TIME_OUT = [(1, 30), (0, 30), (17, 46), (0, 30), (7, 55), (2, 13),
                (0, 30), (11, 46), (0, 30), (9, 54), (0, 30), (9, 51)]
    TIME_IN = [(1, 30), (0, 30), (17, 46), (0, 30), (7, 55), (2, 13),
               (0, 30), (9, 54), (0, 30), (9, 51), (0, 30), (11, 46)]
    INTERVAL_TIME = (0, 30)

    # # PEYI
    # corrected_label_dict = get_corrected_quest_label(
    #     './data/test/label/PEYI.txt', TIME_IN[2:], INTERVAL_TIME, pre_spaces=2)  # task test
    # ecg_info, eda_info = analyze(
    #     './data/test/source/filter/cut/PEYI.txt', inner_draw=False)  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hr_per_minute_info'],
    #               './data/test/label/PEYI/', 'HR')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['frequency'],
    #               './data/test/label/PEYI/', 'HRV_F')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['time'],
    #               './data/test/label/PEYI/', 'HRV_T')
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['non-linear'],
    #               './data/test/label/PEYI/', 'HRV_N')
    # draw_scratter(corrected_label_dict, eda_info['scl_per_minute_info'],
    #               './data/test/label/PEYI/', 'SCL')
    # draw_scratter(corrected_label_dict, {'SCR_TIMES': eda_info['scr_per_minute_info']['times_per_minute']},
    #               './data/test/label/PEYI/', 'SCR_TIMES')
    # draw_scratter(corrected_label_dict, eda_info['scr_per_minute_info']['amplitudes_per_minute_info'],
    #               './data/test/label/PEYI/', 'SCR_AMP')
    # # RZ
    # corrected_label_dict = get_corrected_quest_label(
    #     './data/test/label/RZ.txt', TIME_IN[2:], INTERVAL_TIME, pre_spaces=2)  # task test
    # ecg_info, eda_info = analyze(
    #     './data/test/source/filter/cut/RZ.txt', inner_draw=False)  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hr_per_minute_info'],
    #               './data/test/label/RZ/', 'HR')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['frequency'],
    #               './data/test/label/RZ/', 'HRV_F')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['time'],
    #               './data/test/label/RZ/', 'HRV_T')
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['non-linear'],
    #               './data/test/label/RZ/', 'HRV_N')
    # draw_scratter(corrected_label_dict, eda_info['scl_per_minute_info'],
    #               './data/test/label/RZ/', 'SCL')
    # draw_scratter(corrected_label_dict, {'SCR_TIMES': eda_info['scr_per_minute_info']['times_per_minute']},
    #               './data/test/label/RZ/', 'SCR_TIMES')
    # draw_scratter(corrected_label_dict, eda_info['scr_per_minute_info']['amplitudes_per_minute_info'],
    #               './data/test/label/RZ/', 'SCR_AMP')
    # SYX
    corrected_label_dict = get_corrected_quest_label(
        './data/test/label/SYX.txt', TIME_IN[2:], INTERVAL_TIME, pre_spaces=2)  # task test
    ecg_info, eda_info = analyze(
        './data/test/source/filter/cut/SYX.txt', inner_draw=False)  # task test
    # print(corrected_label_dict['Happy'].size)
    # draw_scratter(corrected_label_dict, ecg_info['hr_per_minute_info'],
    #               './data/test/label/SYX/', 'HR')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['frequency'],
    #               './data/test/label/SYX/', 'HRV_F')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['time'],
    #               './data/test/label/SYX/', 'HRV_T')
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['non-linear'],
    #               './data/test/label/SYX/', 'HRV_N')
    # draw_scratter(corrected_label_dict, eda_info['scl_per_minute_info'],
    #               './data/test/label/SYX/', 'SCL')
    # draw_scratter(corrected_label_dict, {'SCR_TIMES': eda_info['scr_per_minute_info']['times_per_minute']},
    #               './data/test/label/SYX/', 'SCR_TIMES')
    # draw_scratter(corrected_label_dict, eda_info['scr_per_minute_info']['amplitudes_per_minute_info'],
    #               './data/test/label/SYX/', 'SCR_AMP')
    # # Iris
    # corrected_label_dict = get_corrected_quest_label(
    #     './data/test/label/Iris.txt', TIME_OUT[2:], INTERVAL_TIME, pre_spaces=2)  # task test
    # ecg_info, eda_info = analyze(
    #     './data/test/source/filter/cut/Iris.txt', inner_draw=False)  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hr_per_minute_info'],
    #               './data/test/label/Iris/', 'HR')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['frequency'],
    #               './data/test/label/Iris/', 'HRV_F')  # task test
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['time'],
    #               './data/test/label/Iris/', 'HRV_T')
    # draw_scratter(corrected_label_dict, ecg_info['hrv_per_minute_info']['non-linear'],
    #               './data/test/label/Iris/', 'HRV_N')
    # draw_scratter(corrected_label_dict, eda_info['scl_per_minute_info'],
    #               './data/test/label/Iris/', 'SCL')
    # draw_scratter(corrected_label_dict, {'SCR_TIMES': eda_info['scr_per_minute_info']['times_per_minute']},
    #               './data/test/label/Iris/', 'SCR_TIMES')
    # draw_scratter(corrected_label_dict, eda_info['scr_per_minute_info']['amplitudes_per_minute_info'],
    #               './data/test/label/Iris/', 'SCR_AMP')
