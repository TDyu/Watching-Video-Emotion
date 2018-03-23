#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the original data.

Extract the original labchart data into the required files or narray.
"""
from os import listdir
from os.path import isfile
import numpy as np


def extract_single_type(source_path, des_folder_path, signal_type,
                        index=0, max_lines=None):
    """Extract the single signal from output txt data into the required txt data.

    Extract the single signal we need from the original labchart output txt data.
    And then write out the txt data that we will use to analyze.

    Args:
        source_path: str, Path of the original labchart ouput txt data. Need
            filename extension.
        des_folder_path: str, Path of the folder of the extraxted txt data.
        signal_type: str, Type of signal.
        index: int, The number sequence in a line corresponding to the signal
            you want.
        max_lines: int, Max lines data you want to extract. A line means one
            time interval of original data except of NaN.

    Raises:
        FileNotFoundError if the path is wrong.
    """
    file_name = source_path[source_path.rfind(
        '/') + 1 : source_path.find('.txt')]
    des_path = des_folder_path + '/' + file_name + '_' + signal_type + '.txt'
    with open(source_path, 'r', encoding='big5') as source_file:
        with open(des_path, 'w', encoding='utf-8') as des_file:
            line_counting = 0
            for source_line in source_file:
                if source_line[0].isdigit():
                    data_line = source_line.split('\t')
                    signal = data_line[index]
                    if 'NaN' in signal:
                        continue
                    line_counting += 1
                    if index != len(data_line) - 1:
                        signal += '\n'
                    des_file.write(signal)
                    if max_lines and line_counting == max_lines:
                        break


def extract_single_type_narray(source_path, index=0, max_lines=None):
    """Extract the single signal from output txt data into the required narray.

    Extract the single signal we need from the original labchart output txt data.
    And then return narray of float that we will use to analyze.

    Args:
        source_path: str, Path of the original labchart ouput txt data. Need
            filename extension.
        index: int, The number sequence in a line corresponding to the signal
            you want.
        max_lines: int, Max lines data you want to extract. A line means one
            time interval of original data except of NaN.

    Returns:
        The object of (1, max_lines) ndarray of float of extracted signal.

    Raises:
        FileNotFoundError if the path is wrong.
    """
    signals = []
    with open(source_path, 'r', encoding='big5') as source_file:
        line_counting = 0
        for source_line in source_file:
            if source_line[0].isdigit():
                data_line = source_line.split('\t')
                signal = data_line[index]
                if 'NaN' in signal:
                    continue
                line_counting += 1
                signal = float(signal)
                signals.append(signal)
                if max_lines and line_counting == max_lines:
                    break
    return np.array(signals)


def extract_multiple_type(source_path, des_folder_path, signals_type,
                          start_index=0, max_lines=None):
    """Extract the multiple signals from output txt data into the required txt data.

    Extract multiple signals from the original labchart output txt data. And then
    write out the txt data that we will use to analyze. The extracted data
    will add a line header with # to label the signal type of each column. Each
    signal type is separated by a space.

    Args:
        source_path: str, Path of the original labchart ouput txt data. Need
            filename extension.
        des_folder_path: str, Path of the folder of the extraxted txt data.
        signals_type: list of str, Array of the signals type. The number
            sequence of elements correspond to signals in a line.
        start_index: int, The first number sequence in a line corresponding to
            the first signal.
        max_lines: int, Max lines data you want to extract. A line means one
            time interval of original data except of NaN.

    Raises:
        FileNotFoundError if the path is wrong.
    """
    file_name = source_path[source_path.rfind(
        '/') + 1: source_path.find('.txt')]
    des_path = des_folder_path + '/' + file_name + '_multiple.txt'
    with open(source_path, 'r', encoding='big5') as source_file:
        with open(des_path, 'w', encoding='utf-8') as des_file:
            line_counting = 0
            des_file.write('#')
            for signal_type in signals_type:
                if signals_type.index(signal_type) == len(signals_type) - 1:
                    des_file.write(signal_type + '\n')
                else:
                    des_file.write(signal_type + ' ')
            for source_line in source_file:
                if source_line[0].isdigit():
                    signals_line = source_line.split(
                        '\t')[start_index : start_index + len(signals_type)]
                    if 'NaN' in source_line:
                        continue
                    line_counting += 1
                    for signal in signals_line:
                        if signals_line.index(signal) == len(signals_type) - 1 and '\n' in signal:
                            des_file.write(signal)
                        elif signals_line.index(signal) == len(signals_type) - 1:
                            des_file.write(signal + '\n')
                        else:
                            des_file.write(signal + ' ')
                    if max_lines and line_counting == max_lines:
                        break


def extract_multiple_type_narray(source_path, start_index=0, signal_number=1,
                                 max_lines=None):
    """Extract the multiple signals from output txt data into the required txt data.

    Extract multiple signals from the original labchart output txt data. And then
    return narray that we will use to analyze.

    Args:
        source_path: str, Path of the original labchart ouput txt data. Need
            filename extension.
        signals_type: list of str, Array of the signals type. The number
            sequence of elements correspond to signals in a line.
        start_index: int, The first number sequence in a line corresponding to
            the first signal.
        signal_number: int, The number of signals to extract.
        max_lines: int, Max lines data you want to extract. A line means one
            time interval of original data except of NaN.

    Returns:
        The object of (signal_number, max_lines) ndarray of extracted signal.

    Raises:
        FileNotFoundError if the path is wrong.
    """
    signals = []
    with open(source_path, 'r', encoding='big5') as source_file:
        line_counting = 0
        for source_line in source_file:
            if source_line[0].isdigit():
                signals_line = source_line.split(
                    '\t')[start_index : start_index + signal_number]
                if 'NaN' in source_line:
                    continue
                line_counting += 1
                for signal in signals_line:
                    if line_counting == 1:
                        signals.append([float(signal)])
                    else:
                        signals[signals_line.index(signal)].append(
                            float(signal))
                if max_lines and line_counting == max_lines:
                    break
    return np.array(signals)


def extract_all_file(source_folder_path, des_folder_path, signals_type,
                     start_index=0, max_lines=None):
    """Extract the all signals from all output txt data in souce_folder_path
    into the required txt data.

    Extract the all signals from all output txt data in souce_folder_path
    into the required txt data except of copied data with comment. Please note
    that each file in the source_folder_path must have the same signal channel
    sequence.

    Args:
        source_folder_path: str, Path of the folder where has the original
            labchart ouput txt data.
        des_folder_path: str, Path of the folder where has the extraxted txt data.
        signals_type: list of str, Array of the signals type. The number
            sequence of elements correspond to signals in a line.
        start_index: int, The first number sequence in a line corresponding to
            the first signal.
        max_lines: int, Max lines data you want to extract. A line means one
            time interval of original data except of NaN.

    Raises:
        FileNotFoundError if the path is wrong.
    """
    files = listdir(source_folder_path)
    for data_file in files:
        source_path = source_folder_path + '/' + data_file
        if isfile(source_path):
            if 'comment' not in data_file:
                extract_multiple_type(source_path, des_folder_path,
                                      signals_type, start_index, max_lines)


def extract_all_file_divide(source_folder_path, des_folder_path, signals_type,
                            start_index=0, max_lines=None):
    """Extract the all signals from all output txt data in souce_folder_path
    into the each divided required txt data. Each signal is divided into a file.

    Extract the all signals from all output txt data in souce_folder_path
    into the required txt data except of copied data with comment. Each signal
    is divided into a file.Please notice that each file in the source_folder_path
    must have the same signal channel sequence.

    Args:
        source_folder_path: str, Path of the folder where has the original
            labchart ouput txt data.
        des_folder_path: str, Path of the folder where has the extraxted txt data.
        signals_type: list of str, Array of the signals type. The number
            sequence of elements correspond to signals in a line.
        start_index: int, The first number sequence in a line corresponding to
            the first signal.
        max_lines: int, Max lines data you want to extract. A line means one
            time interval of original data except of NaN.

    Raises:
        FileNotFoundError if the path is wrong.
    """
    files = listdir(source_folder_path)
    for data_file in files:
        source_path = source_folder_path + '/' + data_file
        if isfile(source_path):
            if 'comment' not in data_file:
                index = start_index
                for signal_type in signals_type:
                    extract_single_type(source_path, des_folder_path,
                                        signal_type, index, max_lines)
                    index += 1


# if __name__ == '__main__':
#     # extract_all_file('../../data/test/source', '../../../data/test/extracted',
#     #                  ['ECG', 'EDA'], start_index=2) # terminal test
#     # extract_all_file('./data/test/source', './data/test/extracted',
#     #                  ['ECG', 'EDA'], start_index=2) # task test
#     # extract_all_file_divide('./data/test/source', './data/test/extracted',
#     #                         ['ECG', 'EDA'], start_index=2)  # task test
#     # test = extract_single_type_narray(
#     #       './data/test/source/Tie_movie_intro_Zombie.txt', 2) # task test
    # test = extract_multiple_type_narray(
    #     './data/test/source/filter/cut/PEYI.txt', 1, 2) # task test
    # print(test[0])
    # print(test[1])
