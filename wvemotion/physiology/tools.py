#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Tools.

Tools for lab raw data processing.
"""
from sklearn import preprocessing # for normalize
import numpy as np


def normalize(narray):
    """Z-score Normalize all array.

    Args:
        narray: list, narray, Array wantted normalize.
    
    Return:
        A narray representation of normalized array.
    """
    # Correct that input is narray.
    narray = np.array(narray)

    # Correct narray with float data type.
    narray = np.array(narray, dtype=np.float)

    # Normalize.
    return preprocessing.scale(narray)


def normalize_per_type(narray):
    """Normalize each column of narray.

    Args:
        narray: list, narray, Array wantted normalize.
    
    Return:
        A narray representation of normalized array.
    """
    # Correct that input is narray.
    narray = np.array(narray)

    # Correct narray with float data type.
    narray = np.array(narray, dtype=np.float)

    # Each type do normalize.
    type_row_narray = narray.T
    normalize_list = []
    for one_type_row in type_row_narray:
        normalize_list.append(preprocessing.scale(one_type_row))
    normalize_per_type_narray = np.array(normalize_list).T
    return normalize_per_type_narray


def z_score(narray):
    # Correct that input is narray.
    narray = np.array(narray)

    # Correct narray with float data type.
    narray = np.array(narray, dtype=np.float)

    return preprocessing.scale(narray)


def min_max_normalization(narray):
    # Correct that input is narray.
    narray = np.array(narray)

    # Correct narray with float data type.
    narray = np.array(narray, dtype=np.float)

    min_value = np.min(narray)
    max_value = np.max(narray)

    result = []
    for value in narray:
        result.append((value - min_value) / (max_value - min_value))
    
    return np.array(result)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # return min_max_scaler.fit_transform(narray)


# if __name__ == '__main__':
#     test = [[1, 2, 3, 4],
#     [5, 6, 7 ,8]]
#     test = np.array(test, dtype=int)
#     print(normalize(test))
