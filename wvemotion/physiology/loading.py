#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Load phsiology data.
"""
import os
import time

from extraction import extract_multiple_type_narray
from tools import z_score
from tools import min_max_normalization


class LogTime(object):
    """包裝各個時間段的開始與結束，以ms為單位
    """
    def __init__(self, name, start_time, end_time):
        super(LogTime, self).__init__()
        self.name = name
        self.start = start_time
        self.end = end_time
        self.start_ms = self.__to_ms(self.start)
        self.end_ms = self.__to_ms(self.end)

    def __str__(self):
        return 'name: ' + self.name + '\nstart: ' + str(self.start) + '\nend: ' + str(self.end) + '\nstart_ms: ' + str(self.start_ms) + '\nend_ms: ' + str(self.end_ms)


    def __to_ms(self, time_string):
        """H:M:S,f -> ms"""
        time_format = '%Y-%m-%d %H:%M:%S,%f'
        ms_index = time_string.find(',')
        ms = int(time_string[ms_index+1:ms_index+4])
        time_object = time.strptime(time_string, time_format)
        total_ms = (time_object.tm_hour * \
            60 * 60 + time_object.tm_min * \
            60 + time_object.tm_sec) * 1000 + ms
        return total_ms


def cut_log_time_string(line):
    """切出log檔一個line中屬於時間的字串
    """
    post_string = ' [line'
    post_index = line.find(post_string)
    return line[:post_index]


def read_log_time_dict(path, key_list):
    """處理log檔，切出對應時間
    """
    key_time_dict = {}

    with open(path, 'r') as log_file:
        measuring_start = None
        measuring_end = None
        base_start = None
        base_end = None
        current_key = None
        current_start = None
        current_end = None
        for line in log_file:
            if 'Start measuring' in line:
                measuring_start = cut_log_time_string(line)
            elif 'Stop measuring' in line:
                measuring_end  = cut_log_time_string(line)
            elif 'Start base line'in line:
                base_start  = cut_log_time_string(line)
            elif 'End base line'in line:
                base_end  = cut_log_time_string(line)
            elif 'Start play' in line:
                for key in key_list:
                    if key in line:
                        current_key = key
                        break
                current_start = cut_log_time_string(line)
            elif 'End play' in line:
                current_end = cut_log_time_string(line)
                if current_key in line:
                    key_time_dict[current_key] = LogTime(
                        current_key, current_start, current_end)
        key_time_dict['measuring'] = LogTime(
            'measuring', measuring_start, measuring_end)
        key_time_dict['base'] = LogTime('base', base_start, base_end)

    return key_time_dict


class TimePsyData(object):
    """一個時間段所對應的生理量測數據
    """
    def __init__(self, ecg, eda):
        super(TimePsyData, self).__init__()
        self.ecg = ecg
        self.eda = eda


def get_data_narray_dict(key_time_dict, source_path, get_type='original'):
    """讀入生理量測數據后，根據時間段（key_time_dict）做出切割，包裝成TimePsyData
    get_type為標準化方法，只有original（不做），z-score, min-max
    """
    corresponding_data_dict = {}

    data = extract_multiple_type_narray(source_path, 1, 2)
    
    for key, value in key_time_dict.items():
        ecg = data[0][value.start_ms - key_time_dict['measuring']
                      .start_ms: value.end_ms - key_time_dict['measuring'].start_ms]
        eda = data[1][value.start_ms - key_time_dict['measuring']
                      .start_ms: value.end_ms - key_time_dict['measuring'].start_ms]
        if get_type == 'z-score':
            ecg = z_score(ecg)
            eda = z_score(eda)
        elif get_type == 'min-max':
            ecg = min_max_normalization(ecg)
            eda = min_max_normalization(eda)
        corresponding_data_dict[key] = TimePsyData(ecg, eda)
    
    return corresponding_data_dict


class PersonData(object):
    """包裝每一個受測者的各個時間段的量測數據
    """
    def __init__(self, time_name, all_name, source_path, log_path, key_time_dict, corresponding_data_dict):
        super(PersonData, self).__init__()
        self.time_name = time_name
        self.all_name = all_name
        self.source_path = source_path
        self.log_path = log_path
        self.key_time_dict = key_time_dict
        self.corresponding_data_dict = corresponding_data_dict


# def deal_all_people_data(log_parent_path, source_parent_path, key_list):
#     people_dict = {}
#     files = os.listdir(source_parent_path)
#     for f in files:
#         relative_path = os.path.join(source_parent_path, f)
#         if os.path.isfile(relative_path):
#             # Deal path.
#             name_index = f.find('_')
#             file_exten_index = f.find('.txt')
#             time_name = f[:name_index]
#             all_name = f[:file_exten_index]
#             log_path = log_parent_path
#             if log_parent_path[-1] != '/':
#                 log_path += '/' + all_name + 'experiment.log'
#             else:
#                 log_path += all_name + 'experiment.log'
#             source_path = relative_path
#             # Deal time. (Need to deal time and then deal data.)
#             key_time_dict = read_log_time_dict(log_path, key_list)
#             # Deal data.
#             corresponding_data_dict = get_data_narray_dict(
#                 key_time_dict, source_path)
#             # Package.
#             people_dict[time_name] = PersonData(time_name, all_name, source_path, log_path, key_time_dict, corresponding_data_dict)
    
#     return people_dict


# if __name__ == '__main__':
#     key_list = ['4-F', '5-SU', '21-D', '22-A', '28-SA', '35-H']
#     log_parent_path = './data/lab/log/'
#     source_parent_path = './data/lab/source/'
#     people_dict = deal_all_people_data(log_parent_path, source_parent_path, key_list)
