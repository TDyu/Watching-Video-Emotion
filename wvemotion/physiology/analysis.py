#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze phsiology data.

Analyze phsiology data with BioSppy and hrv lib. and do visualization.
"""
from extraction import extract_multiple_type_narray
from plotting import plot_multilayer_histogram
from plotting import plot_multilayer_lind_chart
from tools import normalize_per_type, normalize
from biosppy.signals import ecg
from biosppy.signals import eda
from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear
import numpy as np


def _count_should_label_amount(video_time_list, interval_time=(0, 0)):
    """Calculate should have how many questionnaire labels of these videos.

    Args:
        video_time_list: list of tuple, Time tuples (minute, second) of videos.
        interval_time: tuple, Interval time of main experiment videos.

    Returns:
        A list representation that label should have amount of each main
        experiment video.
    """
    label_amount = []
    for time in video_time_list:
        if time != interval_time:
            if time[1] > 0:
                label_amount.append(time[0] + 1)
            else:
                label_amount.append(time[0])
    return label_amount


def _count_start_end_minute_index(label_amount_list):
    """
    """
    start_end_minute_index_list = []
    pre_amount = 0
    for label_amount in label_amount_list:
        start_end_minute_index_list.append(
            (pre_amount, pre_amount + label_amount))
        pre_amount += label_amount
    return start_end_minute_index_list


def _get_per_video_per_minute_signals(person_per_video_signals_dict, interval_second=0.001):
    """
    """
    per_video_per_minute_signals_dict = {}
    for video_name in person_per_video_signals_dict.keys():
        per_video_per_minute_signals_dict[video_name] = _get_minute_signals(
            person_per_video_signals_dict[video_name], interval_second)
    return per_video_per_minute_signals_dict  # key = video_name, item = narray of narray of signals in a minute


def _arrange_list_by_kind(original_data, according_key, corresponding_time_key,
                     pre_list, kind_list, interval_time=(0, 0), lead_time_number=1):
    """Arrange the previous list of dictionaries.

    Allocated dictionaries which are elements of previous list into new multiple
    dictionaries according to elements of one of original_data key is according_key
    by the amount should be counted by corresponding_time_key of one element of
    original_data.

    Args:
        original_data: list of dict., Original wanted to analyze data.
        according_key: str, Arranged key.
        corresponding_time_key: str, Correspoonding time key of according_key.
        pre_list: list of dict, The list wanted to arrange.
        kind_list: list of str, Keys of inside dict. of arranged list.
            Equivalent to the keys of outside dict. of pre_list.
        interval_time: tuple of int, Interval time of objective of according_key.
        lead_time_number: int, Unnecessary pre-quantity.
    
    Returns:
        A list of dictionary representation of arranged list.
    
    Notes:
        *Dependencies*

        - numpy
    """
    arranged_list = []
    for outer_index in range(0, len(original_data)):
        need_label_amount_list = _count_should_label_amount(
            original_data[outer_index][corresponding_time_key], interval_time)[
                lead_time_number:]
        start_end_minute_index_list = _count_start_end_minute_index(
            need_label_amount_list)
        new_inner_dict = {}
        for video_name in original_data[outer_index][according_key]:
            new_inner_dict[video_name] = {}
            amount_index = original_data[outer_index][according_key].index(
                video_name)
            start_end_minute = start_end_minute_index_list[amount_index]
            for kind_name in kind_list:
                new_inner_dict[video_name][kind_name] = np.array(
                    pre_list[outer_index][kind_name][
                        start_end_minute[0]:start_end_minute[1]])
        arranged_list.append(new_inner_dict)
    return arranged_list


def _arrange_list_to_dict_by_minute(pre_list, outer_key_list, inner_key_list,
                                    calculate_func=np.mean):
    """Arrange the previous list of dictionaries to a dictionary.

    Arrange the list or a dictionary that key is one of outer_key and value is
    dictionary that key is one of inner_key_list and the value is a list of
    statistical value determined by calculate_func calculate all people sorted
    by minute.

    Args:
        pre_list: list of dict, The list wanted to arrange.
        outer_key_list: list of str, Keys of outer dict. of arranged dict.
            Equivalent to the keys of inside dict. of pre_list.
        inner_key_list: list of str, Keys of inner dict. of arranged dict.
            Equivalent to the keys of outside dict. of pre_list.
        kind_list: list of str, Key of inside dict. of arranged list.
            Equivalent to the key of outside dict. of pre_list.
        calculate_func: method of numpy, Method to calculate a statistical value.
    
    Returns:
        A dictionary representation of arranged data.
    
    Notes:
        *Dependencies*

        - numpy
    """
    arranged_dict = {}
    for outer_key in outer_key_list:
        arranged_dict[outer_key] = {}
        for inner_key in inner_key_list:
            arranged_dict[outer_key][inner_key] = []
            for minute_index in range(0, pre_list[0][outer_key][inner_key].size):
                arranged_dict[outer_key][inner_key].append([])
                for one_dict in pre_list:
                    arranged_dict[outer_key][inner_key][minute_index].append(
                        one_dict[outer_key][inner_key][minute_index])
                arranged_dict[outer_key][inner_key][minute_index] = calculate_func(
                    np.array(arranged_dict[outer_key][inner_key][minute_index]))
            arranged_dict[outer_key][inner_key] = np.array(
                arranged_dict[outer_key][inner_key])
    return arranged_dict


def _arrange_dict_by_minute(pre_dict, ori_outer_key_list, ori_inner_key_list):
    """C
    """
    arranged_dict = {}
    for ori_inner_key in ori_inner_key_list:
        arranged_dict[ori_inner_key] = []
        for ori_outer_key in ori_outer_key_list:
            arranged_dict[ori_inner_key].append(pre_dict[ori_outer_key][ori_inner_key])
        arranged_dict[ori_inner_key] = np.array(arranged_dict[ori_inner_key])
    return arranged_dict


def _get_minute_signals(signal, interval_second=0.001):
    """
    """
    minute_amount = int(1 / interval_second * 60)
    enough_minute_times = int(signal.size / 2 / minute_amount)
    minute_signal_list = []
    pre_amount = 0
    amount_list = [(index + 1) * minute_amount for index in range(0, enough_minute_times)]
    for amount in amount_list:
        minute_signal = signal[pre_amount : amount]
        minute_signal_list.append(minute_signal)
        pre_amount = amount
    if signal.size / 2 > pre_amount:
        minute_signal = signal[pre_amount:]
        minute_signal_list.append(minute_signal)
    return minute_signal_list  # list of narray of signals in a video


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
                'SCR_times': output['onsets'].size}
    return eda_info


def get_people_per_video_per_minute_label_dict(
        all_people_data_list, emotion_kind_list,
        interval_time=(0, 0), interval_second=0.001, lead_time_number=1):
    # Get all labels.
    all_labels = []
    for person in all_people_data_list:
        all_labels.append(np.loadtxt(person['label_path']))
    # Each person's label cut section by self watching video order.
    # And normalize the score of same emotion of each video of each person.
    all_label_normalized_per_video_list = []
    for person_index in range(0, len(all_labels)):
        need_label_amount_list = _count_should_label_amount(
            all_people_data_list[person_index]['corresponding_time_list'], interval_time)
        need_label_amount_list = need_label_amount_list[lead_time_number:]
        need_corresponding_index_list = _count_start_end_minute_index(
            need_label_amount_list)
        need_corresponding_index_list = need_corresponding_index_list
        person_dict = {}
        video_index = 0
        for video_name in all_people_data_list[person_index]['video_name_list']:
            start_index = need_corresponding_index_list[video_index][0]
            end_index = need_corresponding_index_list[video_index][1]
            person_dict[video_name] = all_labels[person_index][start_index:end_index]
            person_dict[video_name] = normalize_per_type(
                person_dict[video_name])
            video_index += 1
        all_label_normalized_per_video_list.append(person_dict)
    print(all_label_normalized_per_video_list)
    # Contate labels in same minute of same video of different person.
    # And transpose each same minute matrix.
    # And calculate mean value each same emotion of same minute of same video.
    # Lastly, transepose each same video and arrange to a dict. by video.
    all_label_normalized_per_video_per_minute_dict = {}
    video_name_list = all_people_data_list[0]['video_name_list']
    need_label_amount_list = _count_should_label_amount(
        all_people_data_list[0]['corresponding_time_list'], interval_time)
    need_label_amount_list = need_label_amount_list[lead_time_number:]
    video_index = 0
    for video_name in video_name_list:
        all_label_normalized_per_video_per_minute_dict[video_name] = []
        for minute_index in range(0, need_label_amount_list[video_index]):
            all_label_normalized_per_video_per_minute_dict[video_name].append([])
            kind_per_person_narray = []
            index = 0
            for person_dict in all_label_normalized_per_video_list:
                kind_per_person_narray.append(
                    person_dict[video_name][minute_index])
                index += 1
            kind_per_person_narray = np.array(kind_per_person_narray).T
            for emotion_index in range(0, len(emotion_kind_list)):
                all_label_normalized_per_video_per_minute_dict[video_name][minute_index].append(
                    np.mean(kind_per_person_narray[emotion_index]))
            all_label_normalized_per_video_per_minute_dict[video_name][minute_index] = np.array(
                all_label_normalized_per_video_per_minute_dict[video_name][minute_index])
            kind_per_minute_narray = np.array(all_label_normalized_per_video_per_minute_dict[video_name]).T
        all_label_normalized_per_video_per_minute_dict[video_name] = {}
        emotion_index = 0
        for emotion_kind in emotion_kind_list:
            all_label_normalized_per_video_per_minute_dict[video_name][emotion_kind] = kind_per_minute_narray[emotion_index]
        video_index += 1
    return all_label_normalized_per_video_per_minute_dict


def analyze_video_minute(all_people_data_list, emotion_kind_list,
                         save_folder_path, interval_time=(0, 0),
                         interval_second=0.001, lead_time_number=1):
    # Get dict. that key are video name and value is another dict. that key are
    # emotion kind and value is narray of mean value of all people normalized
    # sorted by minute.
    all_label_normalized_per_video_per_minute_dict = get_people_per_video_per_minute_label_dict(
        all_people_data_list, emotion_kind_list,
        interval_time, interval_second, lead_time_number)
    # Plot
    video_name_list = all_people_data_list[0]['video_name_list']
    corresponding_time_list = all_people_data_list[0]['corresponding_time_list']
    label_should_amount = _count_should_label_amount(
        corresponding_time_list, interval_time)[lead_time_number:]
    color_list = ['red', 'green', 'black', 'yellow', 'blue', 'purple']
    index = 0
    for video in video_name_list:
        if save_folder_path[-1] != '/':
            save_folder_path += '/'
        # histograms
        chart_type = 'histograms'
        save_path = save_folder_path + \
            'per_video/per_minute/emotion_' + chart_type + '/'
        save_name = video + ' - emotion per minute'
        x_axis_list = [index for index in range(
            1, label_should_amount[index] + 1)]
        plot_multilayer_histogram(
            x_axis_list,
            all_label_normalized_per_video_per_minute_dict[video],
            x_axis_list, save_path, color_list, save_name, fig_size=(50, 20),
            bar_width=0.15, labels=False, show=False)
        # lind_chart
        chart_type = 'lind_chart'
        save_path = save_folder_path + \
            'per_video/per_minute/emotion_' + chart_type + '/'
        save_name = video + ' - emotion per minute'
        plot_multilayer_lind_chart(
            x_axis_list,
            all_label_normalized_per_video_per_minute_dict[video],
            x_axis_list, save_path, color_list, save_name,
            fig_size=(20, 20), show=False)
        index += 1
    # # Deal with signals as the following steps.
    # # 1. Read  all signals & Cut
    # all_signals = []
    # for person_dict in all_people_data_list:
    #     signals = extract_multiple_type_narray(
    #         person_dict['signal_path'], 1, 2).T[2 * 60 * 1000:63 * 60 * 1000]
    #     all_signals.append(signals)
    # # 2. Cut per minute
    # all_signals_per_minute_list = []
    # for person in all_signals:
    #     per_minute_signals = _get_minute_signals(person, interval_second)
    #     all_signals_per_minute_list.append(per_minute_signals)
    # # 3. Each minute transpose
    # all_signals_per_minute_tran_list = []
    # for person in all_signals_per_minute_list:
    #     new_minutes_list = []
    #     for minute in person:
    #         new_minutes_list.append(minute.T)
    #     all_signals_per_minute_tran_list.append(np.array(new_minutes_list))
    # # 4. Each minute to calculate
    # all_signals_per_minute_value_list = []
    # physiological_kind_list = ['HR_mean',
    #                            'HR_median', 'HR_variance', 'HR_sd', 'HR_ptp',
    #                            'HRV_lf', 'HRV_lfnu', 'HRV_hf', 'HRV_hfnu', 'HRV_lf_hf',
    #                            'HRV_total_power', 'HRV_vlf',
    #                            'EDA_ptp', 'EDA_diff', 'SCL', 'SCR_times']
    # for person in all_signals_per_minute_tran_list:
    #     physiological_dict = {'HR_mean': [],
    #                           'HR_median': [],
    #                           'HR_variance': [],
    #                           'HR_sd': [],
    #                           'HR_ptp': [],
    #                           'HRV_lf': [],
    #                           'HRV_lfnu': [],
    #                           'HRV_hf': [],
    #                           'HRV_hfnu': [],
    #                           'HRV_lf_hf': [],
    #                           'HRV_total_power': [],
    #                           'HRV_vlf': [],
    #                           'EDA_ptp': [],
    #                           'EDA_diff': [],
    #                           'SCL': [],
    #                           'SCR_times': []}
    #     for minute in person:
    #         ecg_info = ecg_extract(minute[0])
    #         eda_info = eda_extract(minute[1])
    #         for ecg_kind in physiological_kind_list[:12]:
    #             physiological_dict[ecg_kind].append(ecg_info[ecg_kind])
    #         for eda_kind in physiological_kind_list[12:]:
    #             physiological_dict[eda_kind].append(eda_info[eda_kind])
    #     all_signals_per_minute_value_list.append(physiological_dict)
    # # 5. cut video
    # video_name_list = all_people_data_list[0]['video_name_list']
    # all_signals_per_video_per_minute_value_list = _arrange_list_by_kind(
    #     all_people_data_list, 'video_name_list', 'corresponding_time_list',
    #     all_signals_per_minute_value_list, physiological_kind_list,
    #     interval_time, lead_time_number)
    # # 6. median
    # all_signals_median_dict = _arrange_list_to_dict_by_minute(
    #     all_signals_per_video_per_minute_value_list,
    #     video_name_list, physiological_kind_list,
    #     calculate_func=np.median)
    # # 7. plot
    # # corresponding_time_list = all_people_data_list[0]['corresponding_time_list']
    # # label_should_amount = _count_should_label_amount(
    # #     corresponding_time_list, interval_time)[lead_time_number:]
    # # index = 0
    # # for video in video_name_list:
    # #     if save_folder_path[-1] != '/':
    # #         save_folder_path += '/'
    # #     # histograms
    # #     chart_type = 'histograms'
    # #     save_path = save_folder_path + \
    # #         'per_video/per_minute/physiolog_' + chart_type + '/'
    # #     save_name = video + ' - physiology per minute'
    # #     color_list = ['#ea0000', '#004b97', '#01b468', '#930093',
    # #                   '#8600ff', '#d94600', '#a6a600', '#82d900',
    # #                   '#743a3a', '#707038', '#3d7878', '#5151a2',
    # #                   '#7e3d76', '#467500', '#424200', '#7b7b7b']
    # #     x_axis_list = [index for index in range(
    # #         1, label_should_amount[index] + 1)]
    # #     plot_multilayer_histogram(
    # #         x_axis_list,
    # #         all_signals_median_dict[video],
    # #         x_axis_list, save_path, color_list, save_name, size=(50, 20),
    # #         bar_width=0.15, labels=False, show=False)
    # #     # lind_chart
    # #     chart_type = 'lind_chart'
    # #     save_path = save_folder_path + \
    # #         'per_video/per_minute/physiology_' + chart_type + '/'
    # #     save_name = video + ' - physiology per minute'
    # #     plot_multilayer_lind_chart(
    # #         x_axis_list,
    # #         all_signals_median_dict[video],
    # #         x_axis_list, save_path, color_list, save_name, size=(20, 20),
    # #         labels=False, show=False)
    # #     index += 1


def analyze_video_total(all_people_data_list, emotion_kind_list,
                        save_folder_path, interval_time=(0, 0),
                        interval_second=0.001, lead_time_number=1):
    # # Deal with emotion labels as the following steps.
    # # 1. Read all people's labels data and transpose.
    # # 2. Each person create a dictionary according to category of emotion.
    # # 3. Pack all dictionaries into a list.
    # all_labels_per_kind_list = []
    # for person_dict in all_people_data_list:
    #     labels = np.loadtxt(person_dict['label_path']).T
    #     person_kind_dict = {}
    #     for kind_name in emotion_kind_list:
    #         person_kind_dict[kind_name] = labels[
    #             emotion_kind_list.index(kind_name)]
    #     all_labels_per_kind_list.append(person_kind_dict)
    # # 4. Under the above list, each person change to be a dictionary according
    # # to video.
    # all_labels_per_video_list = _arrange_list_by_kind(
    #     all_people_data_list, 'video_name_list', 'corresponding_time_list',
    #     all_labels_per_kind_list, emotion_kind_list,
    #     interval_time, lead_time_number)
    # # 5. Normalize.
    # for person_dict in all_labels_per_video_list:
    #     for video_name in person_dict.keys():
    #         for kind_name in person_dict[video_name].keys():
    #             person_dict[video_name][kind_name] = normalize(
    #                 person_dict[video_name][kind_name])
    # # 6. Connect everyone's data in the same minute & calculate mean value.
    # all_labels_per_video_dict = {}
    # for video_name in all_people_data_list[0]['video_name_list']:
    #     all_labels_per_video_dict[video_name] = {}
    #     for emotion_kind in emotion_kind_list:
    #         all_labels_per_video_dict[video_name][emotion_kind] = []
    #         for person_index in range(0, len(all_people_data_list)):
    #             for minute_index in range(
    #                     0, all_labels_per_video_list[person_index][video_name][emotion_kind].size):
    #                 all_labels_per_video_dict[video_name][emotion_kind].append(
    #                     all_labels_per_video_list[person_index][video_name][emotion_kind][minute_index])
    #         all_labels_per_video_dict[video_name][emotion_kind] = np.mean(np.array(
    #             all_labels_per_video_dict[video_name][emotion_kind]))
    # # 7. Arrange according to kind and sort by minute.
    # all_labels_per_kind_dict = _arrange_dict_by_minute(
    #     all_labels_per_video_dict,
    #     all_people_data_list[0]['video_name_list'], emotion_kind_list)
    # # 8. Plot
    # x_axis_list = np.arange(0, len(emotion_kind_list))
    # x_name = all_people_data_list[0]['video_name_list']
    # # histograms
    # chart_type = 'histograms'
    # save_path = save_folder_path + \
    #     'per_video/unite/emotion_' + chart_type + '/'
    # save_name = 'emotion'
    # color_list = ['red', 'green', 'black', 'yellow', 'blue', 'purple']
    # plot_multilayer_histogram(x_axis_list, all_labels_per_kind_dict, x_name,
    #                           save_path, color_list, save_name, size=(20, 20), bar_width=0.15,
    #                           labels=False, show=False)
    # # lind_chart
    # chart_type = 'lind_chart'
    # save_path = save_folder_path + \
    #     'per_video/unite/emotion_' + chart_type + '/'
    # save_name = 'emotion'
    # color_list = ['red', 'green', 'black', 'yellow', 'blue', 'purple']
    # plot_multilayer_lind_chart(
    #     x_axis_list,
    #     all_labels_per_kind_dict,
    #     x_name, save_path, color_list, save_name, size=(20, 20),
    #     labels=False, show=False)
    # Deal with signals as the following steps.
    pass


def _time_to_signal_index(time_tuple, interval_second=0.001):
    """Calculate the signal index corresponding to the time.
    """
    return int((time_tuple[0] * 60 + time_tuple[1]) * (1 / interval_second))


def _get_corresponding_time_list_signal_dict(signals,
                                             want_time_tuple_list, fragment_kind_list,
                                             interval_second=0.001):
    """Get wantted signl fragments.

    Get signl fragments corresponding to the time in want_time_tuple_list.
    """
    wanted_dict = {}
    for wanted_index in range(0, len(want_time_tuple_list)):
        start_index = _time_to_signal_index(want_time_tuple_list[wanted_index][0])
        end_index = _time_to_signal_index(want_time_tuple_list[wanted_index][1])
        kind = fragment_kind_list[wanted_index]
        wanted_dict[kind] = signals[start_index:end_index]
    return wanted_dict


def get_emotion_fragment_by_value_kind(signal_path, corresponding_time_list,
                                       fragment_kind_list, interval_second=0.001, ecg_normalize=False):
    """
    {psykind: [sorted by fragment order]}
    """
    # Get original signal data.
    signals = extract_multiple_type_narray(signal_path, 1, 2)
    if ecg_normalize:
        signals[1] = normalize(signals[1])
    signals = signals.T
    # Get fragment in a dict with that key are that in fragment_kind_list.
    wantted_signal_dict = _get_corresponding_time_list_signal_dict(
        signals, corresponding_time_list, fragment_kind_list, interval_second)
    # Get ECG and EDA information of each fragment.
    psysiologic_raw_dict = {}
    for fragment_kind, fragment_signal in wantted_signal_dict.items():
        ecg_info_dict = ecg_extract(fragment_signal.T[0])
        eda_info_dict = eda_extract(fragment_signal.T[1])
        psysiologic_raw_dict[fragment_kind] = {'ECG': ecg_info_dict,
                                               'EDA': eda_info_dict}
    # Sort values of each psysiologic kind by order same as wanted kind.
    psysiologic_detail_dict = {}
    physiological_kind_list = ['HR_mean',
                               'HR_median', 'HR_variance', 'HR_sd', 'HR_ptp',
                               'HRV_lf', 'HRV_lfnu', 'HRV_hf', 'HRV_hfnu', 'HRV_lf_hf',
                               'HRV_total_power', 'HRV_vlf',
                               'EDA_ptp', 'EDA_diff', 'SCL', 'SCR_times']
    eda_kind_first_index = 12
    for physiological_kind in physiological_kind_list:
        psysiologic_detail_dict[physiological_kind] = []
        for fragment_kind in fragment_kind_list:
            if physiological_kind in physiological_kind_list[:eda_kind_first_index]:
                psysiologic_detail_dict[physiological_kind].append(
                    psysiologic_raw_dict[fragment_kind]['ECG'][physiological_kind])
            else:
                psysiologic_detail_dict[physiological_kind].append(
                    psysiologic_raw_dict[fragment_kind]['EDA'][physiological_kind])
        psysiologic_detail_dict[physiological_kind] = np.array(
            psysiologic_detail_dict[physiological_kind])
    return psysiologic_detail_dict


def analyze_person_fragment(signal_path, corresponding_time_list,
                             fragment_kind_list, person_name, analysis_type,
                            save_folder_path, interval_second=0.001, ecg_normalize=False):
    """
    For one person.
    {psykind_unit: {psykind: [sorted by fragment order]}}
    """
    # Get dict. that key are value kind and value are narray of  value sorted
    # by fragment order.
    psysiologic_detail_dict = get_emotion_fragment_by_value_kind(
        signal_path, corresponding_time_list, fragment_kind_list,
        interval_second, ecg_normalize)
    # Arrange dict. by different unit.
    psysiologic_detail_unit_dict = {
        'HR': {
            'HR_mean': psysiologic_detail_dict['HR_mean'],
            'HR_median': psysiologic_detail_dict['HR_median'],
            'HR_variance':  psysiologic_detail_dict['HR_variance'],
            'HR_sd':  psysiologic_detail_dict['HR_sd'],
            'HR_ptp':  psysiologic_detail_dict['HR_ptp']},
        'HRV_ms^2': {
            'HRV_lf': psysiologic_detail_dict['HRV_lf'],
            'HRV_hf': psysiologic_detail_dict['HRV_hf'],
            'HRV_vlf': psysiologic_detail_dict['HRV_vlf'],
            'HRV_total_power': psysiologic_detail_dict['HRV_total_power']},
        'HRV_nu': {
            'HRV_lfnu': psysiologic_detail_dict['HRV_lfnu'],
            'HRV_hfnu': psysiologic_detail_dict['HRV_hfnu']},
        'HRV_ratio': {
            'HRV_lf_hf': psysiologic_detail_dict['HRV_lf_hf']},
        'SCL': {
            'EDA_ptp': psysiologic_detail_dict['EDA_ptp'],
            'EDA_diff': psysiologic_detail_dict['EDA_diff'],
            'SCL': psysiologic_detail_dict['SCL']},
        'SCR': {
            'SCR_times': psysiologic_detail_dict['SCR_times']}}
    # Plot
    x_axis_list = np.arange(0, len(fragment_kind_list))
    x_name = fragment_kind_list
    color_dict = {
        'HR': ['red', 'blue', 'green', 'purple', 'yellow'],
        'HRV_ms^2': ['blue', 'red', 'purple', 'yellow'],
        'HRV_nu': ['blue', 'red'],
        'HRV_ratio': ['green'],
        'SCL': ['red', 'blue', 'green'], 'SCR': ['blue']}
    for unit_kind in psysiologic_detail_unit_dict.keys():
        color = color_dict[unit_kind]
        # histograms
        chart_type = 'histograms'
        save_path = save_folder_path + \
            'fragment_analysis_chart/by_person/' + \
            analysis_type + '/' + chart_type + '/'
        save_name = person_name + '_fragment_' + unit_kind
        plot_multilayer_histogram(
            x_axis_list,
            psysiologic_detail_unit_dict[unit_kind],
            x_name, save_path, color, save_name,
            fig_size=(15, 10), font_size=10, bar_width=0.125,
            labels=True, show=False)
        # lind_chart
        chart_type = 'lind_chart'
        save_path = save_folder_path + \
            'fragment_analysis_chart/by_person/' + \
            analysis_type + '/' + chart_type + '/'
        save_name = person_name + '_fragment_' + unit_kind
        plot_multilayer_lind_chart(
            x_axis_list,
            psysiologic_detail_unit_dict[unit_kind],
            x_name, save_path, color, save_name,
            fig_size=(8, 6), font_size=10, show=False)


def analyze_people_fragment(all_people_dict_list, person_name_list,
                            fragment_kind_list, analysis_type, save_folder_path,
                            interval_second=0.001, ecg_normalize=False):
    """
    [{psykind: [sorted by fragment order],...},...]
    => {psykind: {person: [sorted by fragment order]}}
    """
    # Get all people's dict. that key are value kind and value are narray of 
    # value sorted by fragment order.
    all_people_psysiologic_detail_dict_list = []
    for person_dict in all_people_dict_list:
        psysiologic_detail_dict = get_emotion_fragment_by_value_kind(
            person_dict['signal_path'],
            person_dict['fragment_time_tuple'],
            fragment_kind_list,
            interval_second, ecg_normalize)
        all_people_psysiologic_detail_dict_list.append(psysiologic_detail_dict)
    # Arrange to a dict. that key are psysiological kind and the value are
    # dict. that key is people's name and the value is narray of psysiological
    # value sorted by fragment order.
    all_people_psysiologic_dict = {}
    physiological_kind_list = ['HR_mean',
                               'HR_median', 'HR_variance', 'HR_sd', 'HR_ptp',
                               'HRV_lf', 'HRV_lfnu', 'HRV_hf', 'HRV_hfnu', 'HRV_lf_hf',
                               'HRV_total_power', 'HRV_vlf',
                               'EDA_ptp', 'EDA_diff', 'SCL', 'SCR_times']
    for psysiologic_kind in physiological_kind_list:
        all_people_psysiologic_dict[psysiologic_kind] = {}
        person_index = 0
        for person_name in person_name_list:
            all_people_psysiologic_dict[psysiologic_kind][person_name] = []
            for fragment_index in range(0, len(fragment_kind_list)):
                all_people_psysiologic_dict[psysiologic_kind][person_name].append(
                    all_people_psysiologic_detail_dict_list[person_index][psysiologic_kind][fragment_index])
            all_people_psysiologic_dict[psysiologic_kind][person_name] = np.array(
                all_people_psysiologic_dict[psysiologic_kind][person_name])
            person_index += 1
    # Plot
    x_axis_list = np.arange(0, len(fragment_kind_list))
    x_name = fragment_kind_list
    color_list = ['red', 'green', 'purple', 'blue']
    for psysiologic_kind in physiological_kind_list:
        # histograms
        chart_type = 'histograms'
        save_path = save_folder_path + \
            'fragment_analysis_chart/by_kind/' + \
            analysis_type + '/' + chart_type + '/'
        save_name = psysiologic_kind
        plot_multilayer_histogram(
            x_axis_list,
            all_people_psysiologic_dict[psysiologic_kind],
            x_name, save_path, color_list, save_name,
            fig_size=(15, 10), font_size=10, bar_width=0.125,
            labels=True, show=False)
        # lind_chart
        chart_type = 'lind_chart'
        save_path = save_folder_path + \
            'fragment_analysis_chart/by_kind/' + \
            analysis_type + '/' + chart_type + '/'
        save_name = psysiologic_kind
        plot_multilayer_lind_chart(
            x_axis_list,
            all_people_psysiologic_dict[psysiologic_kind],
            x_name, save_path, color_list, save_name,
            fig_size=(8, 6), font_size=10, show=False)


def analyze_person_label_time(label_path, person_name, save_path, need_normalize=False):
    """Total emotion labels line chart per minute.
    """
    label = np.loadtxt(label_path).T
    if need_normalize:
        for singel_emotion in label:
            singel_emotion = normalize(singel_emotion)
        person_name += '_normalized_ver'
    emotion_dict = {
        'angry': label[0],
        'disgusting': label[1],
        'fear': label[2],
        'happy': label[3],
        'sad': label[4],
        'surprise': label[5]}
    x_axis_list = np.arange(1, label[0].size + 1)
    color_list = ['red', 'green', 'black', 'yellow', 'blue', 'purple']
    plot_multilayer_lind_chart(x_axis_list, emotion_dict, x_axis_list,
                               save_path, color_list, person_name,
                               fig_size=(10, 6), font_size=6, show=False)


def analyze_people_label_time(all_label_path_list, people_name_list,
                              save_folder_path, need_normalize=False):
    person_index = 0
    for label_path in all_label_path_list:
        save_path = save_folder_path
        if save_path[-1] != '/':
            save_path += '/'
        save_path += 'labels_analysis_chart/'
        analyze_person_label_time(
            label_path, people_name_list[person_index], save_path, need_normalize)
        person_index += 1


# def analyze_video_minute(all_people_data_list, interval_time=(0, 0),
#                          interval_second=0.001, lead_time_number=1):
#     # Item is dict. (key = video_name, item = narray of labels in the video
#     # (row of narray = time (1 minute), column of narray = emotion type
#     # (Angry, Disgusting, Fear, Happy, Sad, Surprise)))
#     # and items of this list sorted by person order.
#     all_people_per_video_labels_list = []
#     # Item is dict. (key = video_name, item = narray of signals in the video
#     # (row of narray = time (0.001 second), column of narray = signal type (ECG, EDA)))
#     # and items of this list sorted by person order.
#     all_people_per_video_signals_list = []
#     # Item is dict. (key = video_name, item = narray of narray of one minute signals in the video
#     # and items of this list sorted by person order.
#     all_people_per_video_per_minute_signals_list = []
#     # Item is dict. (key = video_name, item = narray of labels in the video
#     # (row of narray = time (1 minute), column of narray = normalized emotion type
#     # (Angry, Disgusting, Fear, Happy, Sad, Surprise)))
#     # and items of this list sorted by person order.
#     all_people_per_video_labels_normalized_list = []
#     for person_data_dict in all_people_data_list:
#         person_per_video_signals_dict = _get_pseson_per_video_signals(
#             person_data_dict['signal_path'],
#             person_data_dict['corresponding_time_list'],
#             person_data_dict['video_name_list'],
#             interval_time,
#             interval_second,
#             lead_time_number)
#         all_people_per_video_signals_list.append(person_per_video_signals_dict)

#         person_per_video_labels_dict = _get_person_per_video_labels(
#             person_data_dict['label_path'],
#             person_data_dict['corresponding_time_list'][lead_time_number + 1:],
#             person_data_dict['video_name_list'],
#             interval_time)
#         all_people_per_video_labels_list.append(person_per_video_labels_dict)

#         per_video_per_minute_signals_dict = _get_per_video_per_minute_signals(
#             person_per_video_signals_dict, interval_second)
#         all_people_per_video_per_minute_signals_list.append(
#             per_video_per_minute_signals_dict)
    
#     # Normalize labels per emotion type.
#     for person_per_video_labels_dict in all_people_per_video_labels_list:
#         person_per_video_labels_normalized_dict = {}
#         for video_name in person_per_video_labels_dict.keys():
#             person_per_video_labels_normalized_dict[video_name] = normalize_per_type(
#                 person_per_video_labels_dict[video_name])
#         all_people_per_video_labels_normalized_list.append(
#             person_per_video_labels_normalized_dict)
#     print(all_people_per_video_labels_normalized_list)


# def _get_pseson_per_video_signals(signal_path, corresponding_time_list,
#                                   video_name_list, interval_time=(0, 0),
#                                   interval_second=0.001, lead_time_number=1):
#     """
#     """
#     # Item is narray that structure is that row = time (0.001 second),
#     # column = signal type (ECG, EDA).
#     signals = extract_multiple_type_narray(signal_path, 1, 2).T
#     accumulated_absolute_time = _get_accumulated_absolute_time(
#         corresponding_time_list,
#         interval_time,
#         interval_second,
#         lead_time_number)
#     return _get_corresponding_signals(signals, accumulated_absolute_time,
#                                       video_name_list, interval_second)


# def _get_accumulated_absolute_time(corresponding_time_list,
#                                    interval_time=(0, 0), interval_second=0.001,
#                                    lead_time_number=1):
#     """
#     """
#     head_tail_time_list = [((0, 0),
#                            (corresponding_time_list[0][0], corresponding_time_list[0][1]))]
#     head_time = head_tail_time_list[0][0]
#     tail_time = head_tail_time_list[0][1]
#     for unit_time in corresponding_time_list[1:]:
#         head_time = (tail_time[0], tail_time[1] + interval_second)
#         tail_time = _find_next_time(tail_time, unit_time)
#         if unit_time != interval_time:
#             head_tail_time_list.append((head_time, tail_time))
#     if lead_time_number > 0:
#         head_tail_time_list = head_tail_time_list[lead_time_number:]
#     return head_tail_time_list  # list of tuple (start_time, end_time) of tuple (minute, second)


# def _get_corresponding_signals(signal, head_tail_time_list,
#                                video_name_list, interval_second=0.001):
#     """
#     """
#     corresponding_signals_dict = {}
#     index = 0
#     for unit_time in head_tail_time_list:
#         corresponding_signals_dict[video_name_list[index]] = signal[
#             _time_to_signal_index(unit_time[0], interval_second) :  _time_to_signal_index(unit_time[1], interval_second) + 1]
#         index += 1
#     return corresponding_signals_dict  # key = video_name, item = narray of signals in this video


# def _find_next_time(pre_time, add_time):
#     """Find the next minute:second time.

#     Args:
#         pre_time: tuple, Previous time (minute, second).
#         add_time: tuple, Add time (minute, second).

#     Returns:
#         A tuple representation of current time (minute, second).
#     """
#     next_time = [pre_time[0] + add_time[0],
#                     pre_time[1] + add_time[1]]
#     if next_time[1] >= 60:
#         next_time[0] += 1
#         next_time[1] -= 60
#     return tuple(next_time)


# def _get_person_per_video_labels(label_path, corresponding_time_list,
#                                  video_name_list, interval_time=(0, 0)):
#     """
#     """
#     person_labels_narray = np.loadtxt(label_path, dtype=int)
#     label_amount_list = _count_should_label_amount(
#         corresponding_time_list, interval_time)
#     labels_per_video_dict = {}
#     pass_amount = 0
#     index = 0
#     for amount in label_amount_list:
#         labels_per_video_list = person_labels_narray[pass_amount : amount + pass_amount]
#         labels_per_video_dict[video_name_list[index]] = labels_per_video_list
#         index += 1
#         pass_amount += amount
#     return labels_per_video_dict  # key = video_name, item = narray of labels in the vide

# ===================================================================

# def analyze(source_path, inner_draw=False):
#     """Extract labchart data and output visual analysis.

#     Extract one person's ECG and EDA data to 2x1 numpy array by
#     extract_multiple_type_narray. (First row is ECG, another row is EDA.)
#     Notice that the 2nd column of original txt data need to be ECG data, and 3rd column need
#     to be EDA data.
#     Afterwards, use ecg_extract and eda_extract to extract the wanted information.
#     At last, draw chart.
#     """
#     signals = extract_multiple_type_narray(source_path, 1, 2)
#     ecg_info = ecg_extract(signals[0], inner_draw=inner_draw)
#     eda_info = eda_extract(signals[1], inner_draw=inner_draw)
#     return ecg_info, eda_info


# def ecg_extract(signal, sampling_rate=1000.0, inner_draw=False):
#     """
#     Args:
#         signal: narray, ECG data.
#         sampling_rate: float, Sampling rate of ECG data.

#     Returns:
#         A dictionary representation of the information of ECG.

#     Notes:
#         *Authors* 
        
#         - the bioSSPy dev team (https://github.com/PIA-Group/BioSPPy)

#         *Dependencies*
        
#         - biosppy
        
#         *See Also*

#         - BioSPPY: https://github.com/PIA-Group/BioSPPy
#     """
#     output = ecg.ecg(signal=signal, sampling_rate=1000.0, show=inner_draw)
#     rr_interval = np.diff(output['rpeaks'])
#     hrv_info, hrv_per_minute_info = get_hrv_all(
#         output['rpeaks'], interval_second=0.001)
#     hr_per_minute_info = _count_hr_per_minute_info(
#         output['heart_rate_ts'], output['heart_rate'])
#     ecg_info = {'signal_time': output['ts'],
#                 'filtered': output['filtered'],
#                 'rpeaks': output['rpeaks'],
#                 'rr_interval': rr_interval,
#                 'heart_rate': output['heart_rate'],
#                 'heart_rate_ts': output['heart_rate_ts'],
#                 'hr_per_minute_info': hr_per_minute_info,
#                 'hrv_time': hrv_info['time'],
#                 'hrv_frequency': hrv_info['frequency'],
#                 'hrv_non-linear': hrv_info['non-linear'],
#                 'hrv_per_minute_info': hrv_per_minute_info}
#     return ecg_info


# def _count_hr_per_minute_info(hr_ts, hr_list):
#     hr_per_minute_info = {}
#     hr_per_minute_list = _count_hr_per_minute(hr_ts, hr_list)
#     hr_per_minute_info['mean'] = _count_mean_per_minute(hr_per_minute_list)
#     hr_per_minute_info['median'] = _count_median_per_minute(
#         hr_per_minute_list)
#     hr_per_minute_info['variance'] = _count_variance_per_minute(
#         hr_per_minute_list)
#     hr_per_minute_info['sd'] = _count_sd_per_minute(hr_per_minute_list)
#     hr_per_minute_info['max_diff'] = _count_max_diff_per_minute(
#         hr_per_minute_list)
#     return hr_per_minute_info


# def _count_hr_per_minute(hr_ts, hr_list):
#     """Cut heart rate into multiple narrays for each minute.

#     Cut heart rate into multiple narrays for each minute. Furthermore, pack into
#     a list (Index is minute).

#     Args:
#         hr_ts: narray, Heart rate time axis reference (seconds).
#         hr_list: narray, Instantaneous heart rate (bpm).
    
#     Returns:
#         A list representation of heart rate values in each minute. Index is minute,
#         and each item is a narray representation of the heart rate in this minute.
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     minute = 1
#     hr_per_minute_list = [[]]
#     for i in range(0, hr_list.size):
#         if hr_ts[i] <= minute * 60.0:
#             pass
#         else:
#             hr_per_minute_list[minute - 1] = np.array(
#                 hr_per_minute_list[minute - 1])
#             hr_per_minute_list.append([])
#             minute += 1
#         hr_per_minute_list[minute - 1].append(hr_list[i])
#     hr_per_minute_list[minute - 1] = np.array(
#         hr_per_minute_list[minute - 1])
#     return hr_per_minute_list


# def _count_mean_per_minute(per_minute_list):
#     """Calculate the mean per minute.

#     Calculate the mean per minute. Furthermore, pack into a list
#     (Index is minute).

#     Args:
#         per_minute_list: list of narray, Physiological values in each minute.
    
#     Returns:
#         A narray representation of mean (Index is minute).
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     mean_per_minute = []
#     for minute_narray in per_minute_list:
#         mean_per_minute.append(np.mean(minute_narray))
#     return np.array(mean_per_minute)


# def _count_median_per_minute(per_minute_list):
#     """Calculate the median per minute.

#     Calculate the median per minute. Furthermore, pack into a list
#     (Index is minute).

#     Args:
#         per_minute_list: list of narray, Physiological values in each minute.
    
#     Returns:
#         A narray representation of median (Index is minute).
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     median_per_minute = []
#     for minute_narray in per_minute_list:
#         median_per_minute.append(np.median(minute_narray))
#     return np.array(median_per_minute)


# def _count_variance_per_minute(per_minute_list):
#     """Calculate the variance per minute.

#     Calculate the variance per minute. Furthermore, pack into a list
#     (Index is minute).

#     Args:
#         per_minute_list: list of narray, Physiological values in each minute.
    
#     Returns:
#         A narray representation of variance (Index is minute).
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     variance_per_minute = []
#     for minute_narray in per_minute_list:
#         variance_per_minute.append(np.var(minute_narray))
#     return np.array(variance_per_minute)


# def _count_sd_per_minute(per_minute_list):
#     """Calculate the standard deviation per minute.

#     Calculate the standard deviation per minute. Furthermore, pack into a list
#     (Index is minute).

#     Args:
#         per_minute_list: list of narray, Physiological values in each minute.
    
#     Returns:
#         A narray representation of standard deviation (Index is minute).
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     sd_per_minute = []
#     for minute_narray in per_minute_list:
#         sd_per_minute.append(np.std(minute_narray))
#     return np.array(sd_per_minute)


# def _count_max_diff_per_minute(per_minute_list):
#     """Calculate the maximum difference per minute.

#     Calculate the maximum difference per minute. Furthermore, pack into a list
#     (Index is minute).

#     Args:
#         per_minute_list: list of narray, Physiological values in each minute.
    
#     Returns:
#         A narray representation of maximum difference (Index is minute).
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     max_diff_per_minute = []
#     for minute_narray in per_minute_list:
#         max_diff_per_minute.append(np.ptp(minute_narray))
#     return np.array(max_diff_per_minute)


# def get_hrv_all(rpeaks, interval_second=0.001):
#     """
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     rrinterval_all = np.diff(rpeaks)
#     hrv_all = get_hrv(rrinterval_all)
#     hrv_per_minute_dict = _count_hrv_per_minute(rpeaks, interval_second)
#     return hrv_all, hrv_per_minute_dict


# def _count_hrv_per_minute(rpeaks, interval_second=0.001):
#     rpeaks_per_minute_list = _count_rpeaks_per_minute(
#         rpeaks, interval_second=0.001)
#     rrinterval_per_minute_list = _count_rrinterval_per_minute(
#         rpeaks_per_minute_list)
#     hrv_per_minute = []
#     hrv_per_minute_dict = {
#         'time': {
#             'mhr': [],
#             'mrri': [],
#             'nn50': [],
#             'pnn50': [],
#             'rmssd': [],
#             'sdnn': []},
#         'frequency': {
#             'hf': [],
#             'hfnu': [],
#             'lf': [],
#             'lf_hf': [],
#             'lfnu': [],
#             'total_power': [],
#             'vlf': []},
#         'non-linear': {
#             'sd1': [],
#             'sd2': []
#         }}
#     for rrinterval in rrinterval_per_minute_list:
#         # UserWarning: nperseg = 256 is greater than input length  = xxx, using nperseg = xxx
#         hrv_per_minute.append(get_hrv(rrinterval))
#     for hvr_info in hrv_per_minute:
#         hrv_per_minute_dict['time']['mhr'].append(
#             hvr_info['time']['mhr'])
#         hrv_per_minute_dict['time']['mrri'].append(
#             hvr_info['time']['mrri'])
#         hrv_per_minute_dict['time']['nn50'].append(
#             hvr_info['time']['nn50'])
#         hrv_per_minute_dict['time']['pnn50'].append(
#             hvr_info['time']['pnn50'])
#         hrv_per_minute_dict['time']['rmssd'].append(
#             hvr_info['time']['rmssd'])
#         hrv_per_minute_dict['time']['sdnn'].append(
#             hvr_info['time']['sdnn'])
#         hrv_per_minute_dict['frequency']['hf'].append(
#             hvr_info['frequency']['hf'])
#         hrv_per_minute_dict['frequency']['hfnu'].append(
#             hvr_info['frequency']['hfnu'])
#         hrv_per_minute_dict['frequency']['lf'].append(
#             hvr_info['frequency']['lf'])
#         hrv_per_minute_dict['frequency']['lf_hf'].append(
#             hvr_info['frequency']['lf_hf'])
#         hrv_per_minute_dict['frequency']['lfnu'].append(
#             hvr_info['frequency']['lfnu'])
#         hrv_per_minute_dict['frequency']['total_power'].append(
#             hvr_info['frequency']['total_power'])
#         hrv_per_minute_dict['frequency']['vlf'].append(
#             hvr_info['frequency']['vlf'])
#         hrv_per_minute_dict['non-linear']['sd1'].append(
#             hvr_info['non-linear']['sd1'])
#         hrv_per_minute_dict['non-linear']['sd2'].append(
#             hvr_info['non-linear']['sd2'])
#     return hrv_per_minute_dict


# def _count_rpeaks_per_minute(rpeaks, interval_second=0.001):
#     """Cut R-Peaks into multiple narrays happened at each minute.

#     Cut appened at multiple narrays happened at each minute. Furthermore, pack
#     into a list (Index is minute).

#     Args:
#         rpeaks, narray, R-peak location indices..
#         interval_second: float, Interval second (1/ Sampling rate).
    
#     Returns:
#         A list representation of R-peak in each minute. Index is minute, and
#         each item is a narray representation of the R-Peaks in this minute.
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     minute = 1
#     rpeaks_per_minute_list = [[]]
#     for i in range(0, rpeaks.size):
#         if rpeaks[i] * interval_second // 60 < minute:
#             pass
#         else:
#             rpeaks_per_minute_list[minute -
#                                    1] = np.array(rpeaks_per_minute_list[minute - 1])
#             rpeaks_per_minute_list.append([])
#             minute += 1
#         rpeaks_per_minute_list[minute - 1].append(rpeaks[i])
#     rpeaks_per_minute_list[minute -
#                            1] = np.array(rpeaks_per_minute_list[minute - 1])
#     return rpeaks_per_minute_list


# def _count_rrinterval_per_minute(rpeaks_per_minute_list):
#     """Calculate RR-interval per minute.

#     Args:
#         rpeaks_per_minute_list: list of narray, R-peaks in each minute.
    
#     Returns:
#         A list of narray, RR-interval per minute.
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     rrinterval_per_minute_list = []
#     for rpeaks in rpeaks_per_minute_list:
#         rrinterval_per_minute_list.append(np.diff(rpeaks))
#     return rrinterval_per_minute_list


# def eda_extract(signal, sampling_rate=1000.0, min_amplitude=0.1, inner_draw=False):
#     """
#     Args:
#         signal: narray, EDA data.
#         sampling_rate: float, Sampling rate of EDA data.
#         min_amplitude: float, Minimum treshold by which to exclude SCRs.

#     Returns:
#         A dictionary representation of the information of EDA.

#     Notes:
#         *Authors* 
        
#         - the bioSSPy dev team (https://github.com/PIA-Group/BioSPPy)

#         *Dependencies*
        
#         - biosppy
        
#         *See Also*
#         - BioSPPY: https://github.com/PIA-Group/BioSPPy
#     """
#     output = eda.eda(signal=signal, sampling_rate=1000.0,
#                      show=inner_draw, min_amplitude=0.1)
#     eda_per_minute_list = _count_eda_per_minute(
#         output['filtered'], interval_second=1 / sampling_rate)
#     scl_per_minute_info = _get_scl_per_minute_info(eda_per_minute_list)
#     scr_per_minute_info = _get_scr_per_minute_info(
#         output['onsets'], output['amplitudes'])
#     eda_info = {'signal_time': output['ts'],
#                 'filtered': output['filtered'],
#                 'onsets': output['onsets'],
#                 'peaks': output['peaks'],
#                 'amplitudes': output['amplitudes'],
#                 'scl_per_minute_info': scl_per_minute_info,
#                 'scr_per_minute_info': scr_per_minute_info}
#     return eda_info


# def _get_scl_per_minute_info(eda_per_minute_list):
#     """Get statistical value of SCL.

#     Args:
#         eda_per_minute_list: list of narray, EDA signals in each minute.
    
#     Returns:
#         A dictionary representation of statistical of SCL.
#     """
#     scl_per_minute_info = {
#         'scl': _count_scl_per_minute(eda_per_minute_list),
#         'max_diff_per_minute': _count_max_diff_per_minute(eda_per_minute_list),
#         'inout_diff_per_minute': _count_eda_inout_diff_per_minute(
#             eda_per_minute_list)
#     }
#     return scl_per_minute_info


# def _get_scr_per_minute_info(onsets, amplitudes):
#     """Get statistical value of SCR.

#     Args:
#         onsets: narray, Indices of SCR pulse onsets.
#         amplitudes: narray, SCR pulse amplitudes.
    
#     Returns:
#         A dictionary representation of statistical of SCR.
#     """
#     times_per_minute = _count_times_scr_per_minute(onsets)
#     scr_per_minute_info = {
#         'times_per_minute':  times_per_minute,
#         'amplitudes_per_minute_info': _get_scr_amplitudes_info_per_minute(
#             times_per_minute, amplitudes)
#     }
#     return scr_per_minute_info
    


# def _count_scl_per_minute(eda_per_minute_list):
#     """Calculate SCL.

#     Args:
#         eda_per_minute_list: list of narray, EDA signals in each minute.
    
#     Returns:
#         A list representation of SCL.
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     scl_per_minute = []
#     for eda_per_minute in eda_per_minute_list:
#         scl_per_minute.append(np.mean(eda_per_minute))
#     return scl_per_minute


# def _count_eda_per_minute(filtered_signal, interval_second=0.001):
#     """Cut filtered EDA signals into multiple narrays for each minute.

#     Args:
#         filtered_signal: narray, Filtered EDA signal.
#         interval_second: float, Interval second (1/ Sampling rate).
    
#     Returns:
#         A narray representation of EDA signals in each minute.
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     minute = 1
#     eda_per_minute_list = [[]]
#     for i in range(0, filtered_signal.size):
#         if (i + 1) * interval_second // 60 < minute:
#             pass
#         else:
#             eda_per_minute_list[minute -
#                                    1] = np.array(eda_per_minute_list[minute - 1])
#             eda_per_minute_list.append([])
#             minute += 1
#         eda_per_minute_list[minute - 1].append(filtered_signal[i])
#     eda_per_minute_list[minute -
#                            1] = np.array(eda_per_minute_list[minute - 1])
#     return eda_per_minute_list


# def _count_eda_inout_diff_per_minute(eda_per_minute_list):
#     """Calculate the value when leaving is subtracted from the value when it comes in.

#     Args:
#         eda_per_minute_list: list of narray, EDA signals in each minute.
    
#     Returns:
#         A list representation of difference that value of leaving minus value of
#         coming in each time.
#     """
#     eda_inout_diff_per_minute = []
#     for eda_per_minute in eda_per_minute_list:
#         eda_inout_diff_per_minute.append(
#             eda_per_minute[-1] - eda_per_minute[1])
#     return eda_inout_diff_per_minute


# def _count_times_scr_per_minute(scr_info, interval_second=0.001):
#     """Calculate SCR times per minute.

#     Args:
#         scr_info: narray, Indices of the SCR peaks or onsets.
#         interval_second: float, Interval second (1/ Sampling rate).

#     Returns:
#         A list representation of times in each minute.
#     """
#     print(scr_info[-1])
#     times_scr_per_minute = [0]
#     minute = 1
#     for info in scr_info:
#         if info * interval_second // 60 < minute:
#             pass
#         else:
#             times_scr_per_minute.append(0)
#             minute += 1
#         times_scr_per_minute[minute - 1] += 1
#     print(len(times_scr_per_minute))
#     print(times_scr_per_minute)
#     return times_scr_per_minute


# def _get_scr_amplitudes_info_per_minute(times_scr_per_minute, amplitudes):
#     """Get multiple statistical values of SCR.

#     Args:
#          times_scr_per_minute: list, SCR times in each minute.
#         amplitudes: narray, SCR pulse amplitudes.
    
#     Returns:
#         A dictionary representation of statistical information of SCR.
#     """
#     amplitudes_per_minute = _count_amplitudes_per_minute(
#         times_scr_per_minute, amplitudes)
#     scr_amplitudes_info = {
#         'mean': _count_mean_per_minute(amplitudes_per_minute),
#         'median': _count_median_per_minute(amplitudes_per_minute),
#         'variance': _count_variance_per_minute(amplitudes_per_minute),
#         'sd': _count_sd_per_minute(amplitudes_per_minute),
#         'max_diff': _count_max_diff_per_minute(amplitudes_per_minute)}
#     return scr_amplitudes_info


# def _count_amplitudes_per_minute(times_scr_per_minute, amplitudes):
#     """Cut SCR amplitudes into multiple narrays for each minute.

#     Args:
#         times_scr_per_minute: list, SCR times in each minute.
#         amplitudes: narray, SCR pulse amplitudes.
    
#     Returns:
#         A list representation of amplitudes in each minute.
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     amplitudes_per_minute = []
#     amplitudes_list = amplitudes.tolist()
#     for i in range(0, len(times_scr_per_minute)):
#         amplitudes_per_minute.append([])
#         for index in [0] * times_scr_per_minute[i]:
#             amplitudes_per_minute[i].append(amplitudes_list.pop(index))
#         amplitudes_per_minute[i] = np.array(amplitudes_per_minute[i])
#     return amplitudes_per_minute


# def reload_label_file(data, data_reload):
#     """Reload questionnaire emotion label file for encoding.
#     """
#     with open(data_reload, 'w', encoding='utf-8') as label_file2:
#         with open(data, 'r', encoding='utf-8') as label_file:
#             for line in label_file:
#                 if '\ufeff' in line:
#                     line = line.replace('\ufeff', '')
#                 label_file2.write(line)


# def get_corrected_quest_label(label_data, video_time_list,interval_time=(0, 0),
#                               pre_spaces=0):
#     """Get scores of emotion labels with corrected amount.

#     Get the scores per minute of each emotion. And the amount of each score need
#     to be able to match with info of heart rate per minute.

#     Args:
#         label_data: str, Path of questionnaire labels.
#         video_time_list: list of tuple, Time tuples (minute, second) of videos.
#         interval_time: tuple, Interval time of main experiment videos.
#         pre_spaces: int, The spaces number during preparation.
    
#     Returns:
#          A dictionary representation of label value of each emotion.

#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     labels = np.loadtxt(label_data, dtype=int)
#     label_should_amount = _count_should_label(video_time_list, interval_time)
#     return _correct_label_value(
#         labels, video_time_list, label_should_amount, interval_time, pre_spaces)


# def _count_should_label(video_time_list, interval_time=(0, 0)):
#     """Calculate should have how many questionnaire labels of these videos.

#     Args:
#         video_time_list: list of tuple, Time tuples (minute, second) of videos.
#         interval_time: tuple, Interval time of main experiment videos.
    
#     Returns:
#         A list representation that label should have amount of each main
#         experiment video.
#     """
#     label_amount = []
#     for time in video_time_list:
#         if time != interval_time:
#             if time[1] > 0:
#                 label_amount.append(time[0] + 1)
#             else:
#                 label_amount.append(time[0])
#     return label_amount


# def _correct_label_value(labels, video_time_list, label_should_amount,
#                          interval_time=(0, 0), pre_spaces=0):
#     """Correct label value to be enough for matching with info of heart rate per minute.

#     Args:
#         labels: list of narray, Questionnaire labels of these videos.
#         video_time_list: list of tuple, Time tuples (minute, second) of videos.
#         label_should_amount: list, Label should have amount of each main
#             experiment video.
#         interval_time: tuple, Interval time of main experiment videos.
#         pre_spaces: int, The spaces number during preparation.
    
#     Returns:
#         A dictionary representation of label value of each emotion.
#     """
#     need_makeup_index = _find_makeup_index(
#         video_time_list, label_should_amount, interval_time)
#     fixed_labels = _fix_label_value(labels, need_makeup_index, pre_spaces)
#     corrected_label_dict = {
#         'Angry': fixed_labels.T[0],
#         'Disgusting': fixed_labels.T[1],
#         'Fear': fixed_labels.T[2],
#         'Happy': fixed_labels.T[3],
#         'Sad': fixed_labels.T[4],
#         'Surprise': fixed_labels.T[5]}
#     return corrected_label_dict


# def _find_makeup_index(video_time_list, label_should_amount, interval_time=(0, 0)):
#     """Find the index of the points which need to make up.

#     Args:
#         video_time_list: list of tuple, Time tuples (minute, second) of videos.
#         label_should_amount: list, Label should have amount of each main
#             experiment video.
#         interval_time: tuple, Interval time of main experiment videos.
    
#     Returns:
#         A list representation of the index of the points which needed to make up.
#     """
#     need_makeup_index = []
#     current_time = [video_time_list[0][0], video_time_list[0][1]]
#     mackup_amount = 0
#     interval_times = 0
#     for i in range(1, len(video_time_list)):
#         if i != len(video_time_list) - 1 and video_time_list[i] == interval_time:
#             pre_time = current_time
#             current_time = current_time = _find_next_time(
#                 current_time, video_time_list[i])
#             if current_time[0] == pre_time[0]:

#                 # "i" is the index which "video" need to make up.
#                 # In order to obtain the correct accumulated ceiling index value
#                 # of label_should_amount, i need to minus 1 because that this i
#                 # is from 1, and then need to minus interval_times to pass index
#                 # the video which are interval videos. After accumulating the
#                 # number of labels that should have passed, need to add number
#                 # numbar generated due to compensation.

#                 need_makeup_index.append(
#                     sum(label_should_amount[:i - 1 - interval_times]) + mackup_amount)
#                 mackup_amount += 1
#         else:
#             current_time = _find_next_time(current_time, video_time_list[i])
#         if interval_time != (0, 0) and video_time_list[i] == interval_time and i != 1:
#             interval_times += 1
#     return need_makeup_index


# def _find_next_time(pre_time, add_time):
#     """Find the next minute:second time.

#     Args:
#         pre_time: tuple, Previous time (minute, second).
#         add_time: tuple, Add time (minute, second).
    
#     Returns:
#         A tuple representation of current time (minute, second).
#     """
#     next_time = [pre_time[0] + add_time[0],
#                     pre_time[1] + add_time[1]]
#     if next_time[1] >= 60:
#         next_time[0] += 1
#         next_time[1] -= 60
#     return next_time


# def _fix_label_value(labels, need_makeup_index, pre_spaces=0):
#     """Make up the label value with 0.

#     Args:
#         labels: list of narray, Questionnaire labels of these videos.
#         need_makeup_index: list, The index of the points which needed to make up.
#         pre_spaces: int, The spaces number during preparation.
    
#     Returns:
#         A list of narray representation of maken up label value that can use to
#         match heart rate info per minute.
    
#     Notes:
#         *Dependencies*
        
#         - numpy
#     """
#     fixed_labels = labels
#     make_up = [0, 0, 0, 0, 0, 0]
#     for index in need_makeup_index:
#         fixed_labels = np.insert(fixed_labels, index, values=make_up, axis=0)
#     if pre_spaces != 0:
#         for index in [0] * pre_spaces:
#             fixed_labels = np.insert(
#                 fixed_labels, index, values=make_up, axis=0)
#     return fixed_labels


# def draw_scratter(corrected_label_dict, per_minute_info, save_folder, phy_type):
#     """Draw scratter chart with physiological information and questionnaire
#     emotion labels.

#     Draw scratter chart. X axis is one type physiological data. Y axis is one
#     type questionnaire emotion labels.

#     Args:
#         corrected_label_dict: dict of narray, Corrected label value of each emotion.
#         per_minute_info: dict of narray, Information per minute.
#         sace_folder: str, Specified image storage path.
#         phy_type: str, Type of physiological information (HR, HRV_F, HRV_T, HRV_N,...).
#             HR => ecg_info['hr_per_minute_info']
#             HRV_F => ecg_info['hrv_per_minute_info']['frequency']
#             HRV_T => ecg_info['hrv_per_minute_info']['time']
#             HRV_N => ecg_info['hrv_per_minute_info']['non-linear']
#             SCL => eda_info['scl_per_minute_info']
#             SCR_TIMES => {'SCR_TIMES': eda_info['scr_per_minute_info']['times_per_minute']}
#             SCR_AMP => eda_info['scr_per_minute_info']['amplitudes_per_minute_info']

#     Notes:
#         *Dependencies*
        
#         - matplotlib
#     """
#     COLOR = {
#         'Angry': 'red',
#         'Disgusting': 'pink',
#         'Fear': 'black',
#         'Happy': 'yellow',
#         'Sad': 'blue',
#         'Surprise': 'purple'}
#     save_folder_path = save_folder + phy_type + '/'
#     if not os.path.exists(save_folder_path):
#         os.makedirs(save_folder_path)
#     for info in per_minute_info.keys():
#         for emotion in corrected_label_dict.keys():
#             img_name = save_folder_path + info + '_' + emotion + '.png'
#             plt.xlabel(info + ' of ' + phy_type + '(minute)')
#             plt.ylabel('scores of ' + emotion + ' (minute)')
#             plt.scatter(per_minute_info[info], corrected_label_dict[emotion],
#                         c=COLOR[emotion], s=25, alpha=1.0, marker='o')
#             plt.savefig(img_name)
#             plt.clf()


if __name__ == '__main__':
    LABELS_PATH = ['./data/test/label/PEYI.txt',
            './data/test/label/RZ.txt',
            './data/test/label/SYX.txt',
            './data/test/label/Iris.txt']
    EMOTION_KIND = ['angry', 'disgusting', 'fear', 'happy', 'sad', 'surprise']
    SIGNALS_PATH = ['./data/test/source/filter/cut/PEYI.txt',
                   './data/test/source/filter/cut/RZ.txt',
                   './data/test/source/filter/cut/SYX.txt',
                   './data/test/source/filter/cut/Iris.txt']
    VIDEO_NAME_IN = ['QGZ', 'HM', 'YEF', 'GY', 'CURVE', 'KX']
    VIDEO_NAME_OUT = ['QGZ', 'HM', 'YEF', 'KX', 'GY', 'CURVE']
    TIME_OUT = [(1, 30), (0, 30), (17, 46), (0, 30), (7, 55), (2, 13),
                (0, 30), (11, 46), (0, 30), (9, 54), (0, 30), (9, 51)]
    TIME_IN = [(1, 30), (0, 30), (17, 46), (0, 30), (7, 55), (2, 13),
               (0, 30), (9, 54), (0, 30), (9, 51), (0, 30), (11, 46)]
    PEYI_DATA = {'signal_path': SIGNALS_PATH[0], 'label_path': LABELS_PATH[0],
            'corresponding_time_list': TIME_IN, 'video_name_list': VIDEO_NAME_IN}
    RZ_DATA = {'signal_path': SIGNALS_PATH[1], 'label_path': LABELS_PATH[1],
               'corresponding_time_list': TIME_IN, 'video_name_list': VIDEO_NAME_IN}
    SYX_DATA = {'signal_path': SIGNALS_PATH[2], 'label_path': LABELS_PATH[2],
                'corresponding_time_list': TIME_IN, 'video_name_list': VIDEO_NAME_IN}
    IRIS_DATA = {'signal_path': SIGNALS_PATH[3], 'label_path': LABELS_PATH[3],
                 'corresponding_time_list': TIME_OUT, 'video_name_list': VIDEO_NAME_OUT}
    # Fragment need to consider lead time.
    IRIS_EMOTION_FRAGMENT = (
        ((33, 0), (48, 30)), ((54, 0), (63, 0)), ((28, 30), (32, 30)))
    PEYI_EMOTION_FRAGMENT = (
        ((52, 0), (63, 0)), ((30, 0), (51, 0)), ((40, 0), (52, 0)))
    SYX_EMOTION_FRAGMENT = (
        ((52, 0), (63, 0)), ((29, 0), (43, 0)), ((28, 0), (41, 0)))
    RZ_EMOTION_FRAGMENT = (
        ((52, 0), (63, 0)), ((29, 0), (51, 0)), ((28, 0), (43, 0)))
    SPECIFIC_VIDEO_FRAGMENT_IN = (((51, 39), (63, 25)), ((30, 54), (40, 48)))
    SPECIFIC_VIDEO_FRAGMENT_OUT = (((34, 54), (42, 40)), ((53, 34), (63, 25)))
    SPECIFIC_VIDEO_FRAGMENT_KIND_LIST = ['KX(happy)', 'CURVE(fear)']
    EMOTION_FRAGMENT_KIND_LIST = ['happy', 'fear', 'surprise']
    KX_FRAGMENT_OUT = (
        ((30, 54), (32, 54)),
        ((32, 54), (34, 54)),
        ((34, 54), (36, 54)),
        ((36, 54), (38, 54)),
        ((38, 54), (42, 40)))
    KX_FRAGMENT_IN = (
        ((51, 39), (53, 39)),
        ((53, 39), (55, 39)),
        ((55, 39), (57, 39)),
        ((57, 39), (59, 39)),
        ((59, 39), (63, 25)))
    CURVE_FRAGMENT_OUT = (
        ((53, 34), (55, 34)),
        ((55, 34), (57, 34)),
        ((57, 34), (59, 34)),
        ((59, 34), (61, 34)),
        ((61, 34), (63, 25)))
    CURVE_FRAGMENT_IN = (
        ((41, 18), (43, 18)),
        ((43, 18), (45, 18)),
        ((45, 18), (47, 18)),
        ((47, 18), (49, 18)),
        ((49, 18), (51, 9)))
    KX_FRAGMENT_SLIDING_OUT = (
        ((30, 54), (35, 54)),
        ((32, 54), (37, 54)),
        ((33, 54), (38, 54)),
        ((34, 54), (39, 54)),
        ((35, 54), (42, 40)))
    KX_FRAGMENT_SLIDING_IN = (
        ((51, 39), (56, 39)),
        ((53, 39), (58, 39)),
        ((54, 39), (59, 39)),
        ((55, 39), (60, 39)),
        ((56, 39), (63, 25)))
    CURVE_FRAGMENT_SLIDING_OUT = (
        ((53, 34), (58, 34)),
        ((55, 34), (60, 34)),
        ((56, 34), (61, 34)),
        ((57, 34), (62, 34)),
        ((58, 34), (63, 25)))
    CURVE_FRAGMENT_SLIDING_IN = (
        ((41, 18), (46, 18)),
        ((43, 18), (48, 18)),
        ((44, 18), (49, 18)),
        ((45, 18), (50, 18)),
        ((46, 18), (51, 9)))
    GY_POINT_FRAGMENT_OUT = (
        ((46, 24), (46, 44)),
        ((50, 53), (51, 13)),
        ((51, 37), (51, 57)),
        ((52, 45), (53, 5)))
    GY_POINT_FRAGMENT_IN = (
        ((34, 8), (34, 28)),
        ((38, 37), (38, 57)),
        ((39, 21), (39, 41)),
        ((40, 29), (40, 49)))
    GY_POINT_KIND_LIST = ['1st', '2nd', '3rd', '4th']
    SIGNAL_VIDEO_FRAGMENT_KIND_LIST = ['2min', '4min', '6min', '8min', 'END']
    SIGNAL_VIDEO_FRAGMENT_SLIDING_KIND_LIST = [
        '0-5min', '2-7min', '3-8min', '4-9min', '5-END']
    PEOPLE_NAME_LIST = ['PEYI', 'RZ', 'SYX', 'IRIS']
    ALL_DATA= [PEYI_DATA, RZ_DATA, SYX_DATA, IRIS_DATA]
    INTERVAL_TIME = (0, 30)
    INTERVAL_SECOND = 0.001
    LEAD_TIME_NUMBER = 1
    SAVE_FOLDER_ROOT_PATH = './data/test/label/'
    # # emotion analysis fragment (per person per value kind per fragment) (check.)
    # analyze_person_fragment(PEYI_DATA['signal_path'], PEYI_EMOTION_FRAGMENT,
    #                         EMOTION_FRAGMENT_KIND_LIST, 'PEYI', 'emotion_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # analyze_person_fragment(RZ_DATA['signal_path'], RZ_EMOTION_FRAGMENT,
    #                         EMOTION_FRAGMENT_KIND_LIST, 'RZ', 'emotion_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # analyze_person_fragment(SYX_DATA['signal_path'], SYX_EMOTION_FRAGMENT,
    #                         EMOTION_FRAGMENT_KIND_LIST, 'SYX', 'emotion_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # analyze_person_fragment(IRIS_DATA['signal_path'], IRIS_EMOTION_FRAGMENT,
    #                         EMOTION_FRAGMENT_KIND_LIST, 'IRIS', 'emotion_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    
    # # video analysis fragment (per person per value kind per fragment) (check.)
    # analyze_person_fragment(PEYI_DATA['signal_path'], SPECIFIC_VIDEO_FRAGMENT_IN,
    #                         SPECIFIC_VIDEO_FRAGMENT_KIND_LIST, 'PEYI', 'video_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # analyze_person_fragment(RZ_DATA['signal_path'], SPECIFIC_VIDEO_FRAGMENT_IN,
    #                         SPECIFIC_VIDEO_FRAGMENT_KIND_LIST, 'RZ', 'video_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # analyze_person_fragment(SYX_DATA['signal_path'], SPECIFIC_VIDEO_FRAGMENT_IN,
    #                         SPECIFIC_VIDEO_FRAGMENT_KIND_LIST, 'SYX', 'video_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # analyze_person_fragment(IRIS_DATA['signal_path'], SPECIFIC_VIDEO_FRAGMENT_OUT,
    #                         SPECIFIC_VIDEO_FRAGMENT_KIND_LIST, 'IRIS', 'video_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)

    # # video analysis fragment (per value kind per person per fragment)
    # PEYI_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[0],
    #     'fragment_time_tuple': SPECIFIC_VIDEO_FRAGMENT_IN}
    # RZ_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[1],
    #     'fragment_time_tuple': SPECIFIC_VIDEO_FRAGMENT_IN}
    # SYX_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[2],
    #     'fragment_time_tuple': SPECIFIC_VIDEO_FRAGMENT_IN}
    # IRIS_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[3],
    #     'fragment_time_tuple': SPECIFIC_VIDEO_FRAGMENT_OUT}
    # PEOPLE_NAME_LIST = ['PEYI', 'RZ', 'SYX', 'IRIS']
    # all_video_analysis_dict_list = [
    #     PEYI_FRAGMENT_DATA,
    #     RZ_FRAGMENT_DATA,
    #     SYX_FRAGMENT_DATA,
    #     IRIS_FRAGMENT_DATA]
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         SPECIFIC_VIDEO_FRAGMENT_KIND_LIST, 'video_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    
    # # KX analysis fragment (per value kind per person per fragment) (check)
    # PEYI_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[0],
    #     'fragment_time_tuple': KX_FRAGMENT_IN}
    # RZ_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[1],
    #     'fragment_time_tuple': KX_FRAGMENT_IN}
    # SYX_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[2],
    #     'fragment_time_tuple': KX_FRAGMENT_IN}
    # IRIS_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[3],
    #     'fragment_time_tuple': KX_FRAGMENT_OUT}
    # all_video_analysis_dict_list = [
    #     PEYI_FRAGMENT_DATA,
    #     RZ_FRAGMENT_DATA,
    #     SYX_FRAGMENT_DATA,
    #     IRIS_FRAGMENT_DATA]
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         SIGNAL_VIDEO_FRAGMENT_KIND_LIST, 'KX_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # # normalized ecg ver
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         SIGNAL_VIDEO_FRAGMENT_KIND_LIST, 'KX_fragment_normalized_ecg_ver',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND, ecg_normalize=True)
    # # CURVE analysis fragment (per value kind per person per fragment)  (check)
    # PEYI_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[0],
    #     'fragment_time_tuple': CURVE_FRAGMENT_IN}
    # RZ_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[1],
    #     'fragment_time_tuple': CURVE_FRAGMENT_IN}
    # SYX_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[2],
    #     'fragment_time_tuple': CURVE_FRAGMENT_IN}
    # IRIS_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[3],
    #     'fragment_time_tuple': CURVE_FRAGMENT_OUT}
    # PEOPLE_NAME_LIST = ['PEYI', 'RZ', 'SYX', 'IRIS']
    # all_video_analysis_dict_list = [
    #     PEYI_FRAGMENT_DATA,
    #     RZ_FRAGMENT_DATA,
    #     SYX_FRAGMENT_DATA,
    #     IRIS_FRAGMENT_DATA]
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         SIGNAL_VIDEO_FRAGMENT_KIND_LIST, 'CURVE_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # # normalize ver
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         SIGNAL_VIDEO_FRAGMENT_KIND_LIST, 'CURVE_fragment_normalized_ecg_ver',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND, ecg_normalize=True)

    # # GY analysis fragment (per value kind per person per fragment) (check)
    # PEYI_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[0],
    #     'fragment_time_tuple': GY_POINT_FRAGMENT_IN}
    # RZ_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[1],
    #     'fragment_time_tuple': GY_POINT_FRAGMENT_IN}
    # SYX_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[2],
    #     'fragment_time_tuple': GY_POINT_FRAGMENT_IN}
    # IRIS_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[3],
    #     'fragment_time_tuple': GY_POINT_FRAGMENT_OUT}
    # all_video_analysis_dict_list = [
    #     PEYI_FRAGMENT_DATA,
    #     RZ_FRAGMENT_DATA,
    #     SYX_FRAGMENT_DATA,
    #     IRIS_FRAGMENT_DATA]
    # # normalized ecg ver
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         GY_POINT_KIND_LIST, 'GY_fragment_normalized_ver',
    #                         SAVE_FOLDER_ROOT_PATH, ecg_normalize=True)

    # # people label per minute  (check.)
    # analyze_people_label_time(LABELS_PATH, PEOPLE_NAME_LIST,
    #                           SAVE_FOLDER_ROOT_PATH)
    # # normalize ver
    # analyze_people_label_time(LABELS_PATH, PEOPLE_NAME_LIST,
    #                           SAVE_FOLDER_ROOT_PATH, need_normalize=True)

    # # KX sliding window analysis fragment (per value kind per person per fragment)  (check.)
    # PEYI_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[0],
    #     'fragment_time_tuple': KX_FRAGMENT_SLIDING_IN}
    # RZ_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[1],
    #     'fragment_time_tuple': KX_FRAGMENT_SLIDING_IN}
    # SYX_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[2],
    #     'fragment_time_tuple': KX_FRAGMENT_SLIDING_IN}
    # IRIS_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[3],
    #     'fragment_time_tuple': KX_FRAGMENT_SLIDING_OUT}
    # all_video_analysis_dict_list = [
    #     PEYI_FRAGMENT_DATA,
    #     RZ_FRAGMENT_DATA,
    #     SYX_FRAGMENT_DATA,
    #     IRIS_FRAGMENT_DATA]
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         SIGNAL_VIDEO_FRAGMENT_SLIDING_KIND_LIST, 'KX_sliding_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    # # CURVE analysis fragment (per value kind per person per fragment)  (check)
    # PEYI_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[0],
    #     'fragment_time_tuple': CURVE_FRAGMENT_SLIDING_IN}
    # RZ_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[1],
    #     'fragment_time_tuple': CURVE_FRAGMENT_SLIDING_IN}
    # SYX_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[2],
    #     'fragment_time_tuple': CURVE_FRAGMENT_SLIDING_IN}
    # IRIS_FRAGMENT_DATA = {
    #     'signal_path': SIGNALS_PATH[3],
    #     'fragment_time_tuple': CURVE_FRAGMENT_SLIDING_OUT}
    # PEOPLE_NAME_LIST = ['PEYI', 'RZ', 'SYX', 'IRIS']
    # all_video_analysis_dict_list = [
    #     PEYI_FRAGMENT_DATA,
    #     RZ_FRAGMENT_DATA,
    #     SYX_FRAGMENT_DATA,
    #     IRIS_FRAGMENT_DATA]
    # analyze_people_fragment(all_video_analysis_dict_list, PEOPLE_NAME_LIST,
    #                         SIGNAL_VIDEO_FRAGMENT_SLIDING_KIND_LIST, 'CURVE_sliding_fragment',
    #                         SAVE_FOLDER_ROOT_PATH, INTERVAL_SECOND)
    pass

    # per video per minute
    # analyze_video_minute(ALL_DATA, EMOTION_KIND, './data/test/label/',
    #                      INTERVAL_TIME, INTERVAL_SECOND, LEAD_TIME_NUMBER)
    # analyze_video_total(ALL_DATA, EMOTION_KIND, './data/test/label/',
    #                      INTERVAL_TIME, INTERVAL_SECOND, LEAD_TIME_NUMBER)

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
    # # SYX
    # corrected_label_dict = get_corrected_quest_label(
    #     './data/test/label/SYX.txt', TIME_IN[2:], INTERVAL_TIME, pre_spaces=2)  # task test
    # ecg_info, eda_info = analyze(
    #     './data/test/source/filter/cut/SYX.txt', inner_draw=False)  # task test
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
    # draw_scratter(corrected_label_, 2, dict, eda_info['scl_per_minute_info'],
    #               './data/test/label/Iris/', 'SCL')
    # draw_scratter(corrected_label_dict, {'SCR_TIMES': eda_info['scr_per_minute_info']['times_per_minute']},
    #               './data/test/label/Iris/', 'SCR_TIMES')
    # draw_scratter(corrected_label_dict, eda_info['scr_per_minute_info']['amplitudes_per_minute_info'],
    #               './data/test/label/Iris/', 'SCR_AMP')
