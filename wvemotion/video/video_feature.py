#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Extract video features.
http://monkeycoding.com/?p=690#OpenCV
https://blog.csdn.net/weiweigfkd/article/details/20898937
"""
import time
import datetime
import copy
from functools import reduce
import math
import os
import operator

from cv2 import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def _base_key_dict_generator(name, array, need_differ=True):
    """組成字典表示數值，以下key為後綴，前綴又name做決定
    """
    array = np.array(array)
    lower_quartile, upper_quartile, tenth_quartile, ninetieth_quartile = np.percentile(
        array, [25, 75, 10, 90])
    key_dict = {
        name + '_mean': np.mean(array),
        name + '_median': np.median(array),
        name + '_var': np.var(array),
        name + '_sd': np.std(array),
        name + '_max': np.max(array),
        name + '_min': np.min(array),
        name + '_lower_quartile': lower_quartile,
        name + '_upper_quartile': upper_quartile,
        name + '_10th_percentile': tenth_quartile,
        name + '_90th_percentile': ninetieth_quartile
    }
    if need_differ:
        differs_array = np.diff(array)
        lower_quartile, upper_quartile, tenth_quartile, ninetieth_quartile = np.percentile(
            differs_array, [25, 75, 10, 90])
        key_dict[name + '_differ_mean'] = np.mean(differs_array)
        key_dict[name + '_differ_median'] = np.median(differs_array)
        key_dict[name + '_differ_var'] = np.var(differs_array)
        key_dict[name + '_differ_sd'] = np.std(differs_array)
        key_dict[name + '_differ_max'] = np.max(differs_array)
        key_dict[name + '_differ_min'] = np.min(differs_array)
        key_dict[name + '_differ_lower_quartile'] = lower_quartile
        key_dict[name + '_differ_upper_quartile'] = upper_quartile
        key_dict[name + '_differ_10th_percentile'] = tenth_quartile
        key_dict[name + '_differ_90th_percentile'] = ninetieth_quartile
    return key_dict


def _compare_rgb_rms(picture1, picture2):
    """Compare histogram RMS of rgb between two image.
    """
    image1 = Image.open(picture1)
    image2 = Image.open(picture2)

    histogram1 = image1.histogram()
    histogram2 = image2.histogram()

    differ = math.sqrt(reduce(operator.add, list(
        map(lambda a, b: (a - b) ** 2, histogram1, histogram2))) / len(histogram1))

    return differ


def rgb_differ(images_path_list):
    """Compare a video RMS similarity by each second frame.
    """
    differs = []
    index = 1
    for image_path in images_path_list[1:]:
        differs.append(_compare_rgb_rms(
            images_path_list[index - 1], image_path))
        index += 1
    differs = np.array(differs)
    return _base_key_dict_generator('rgb_differ', differs, need_differ=False)


def _extract_rgb_value(image_path):
    image = cv2.imread(image_path)
    # 要注意opencv讀出來的RGB通道順序是GBR
    b, g, r = cv2.split(image)
    return np.mean(r),  np.mean(g),  np.mean(b)


def rgb_value(images_path_list):
    r_list = []
    g_list = []
    b_list = []
    for image_path in images_path_list:
        r, g, b = _extract_rgb_value(image_path)
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
    rgb_value_dict = _base_key_dict_generator(
        'rgb_r_value', r_list, need_differ=True)
    rgb_value_dict.update(_base_key_dict_generator(
        'rgb_g_value', g_list, need_differ=True))
    rgb_value_dict.update(_base_key_dict_generator(
        'rgb_b_value', b_list, need_differ=True))
    return rgb_value_dict


def _extract_hsv_value(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return np.mean(h), np.mean(s), np.mean(v)


def hsv_value(images_path_list):
    h_list = []
    s_list = []
    v_list = []
    for image_path in images_path_list:
        h, s, v = _extract_hsv_value(image_path)
        h_list.append(h)
        s_list.append(s)
        v_list.append(v)
    hsv_value_dict = _base_key_dict_generator(
        'hsv_h_value', h_list, need_differ=True)
    hsv_value_dict.update(_base_key_dict_generator(
        'hsv_s_value', s_list, need_differ=True))
    hsv_value_dict.update(_base_key_dict_generator(
        'hsv_v_value', v_list, need_differ=True))
    return hsv_value_dict


def _extract_hsl_value(image_path):
    image = cv2.imread(image_path)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    return np.mean(h), np.mean(s), np.mean(l)


def hsl_value(images_path_list):
    h_list = []
    s_list = []
    l_list = []
    for image_path in images_path_list:
        h, s, l = _extract_hsl_value(image_path)
        h_list.append(h)
        s_list.append(s)
        l_list.append(l)
    hsl_value_dict = _base_key_dict_generator(
        'hsl_h_value', h_list, need_differ=True)
    hsl_value_dict.update(_base_key_dict_generator(
        'hsl_s_value', s_list, need_differ=True))
    hsl_value_dict.update(_base_key_dict_generator(
        'hsl_l_value', l_list, need_differ=True))
    return hsl_value_dict


def _extract_Lab_value(image_path):
    image = cv2.imread(image_path)
    Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(Lab)
    return np.mean(L), np.mean(a), np.mean(b)


def Lab_value(images_path_list):
    L_list = []
    a_list = []
    b_list = []
    for image_path in images_path_list:
        L, a, b = _extract_Lab_value(image_path)
        L_list.append(L)
        a_list.append(a)
        b_list.append(b)
    Lab_value_dict = _base_key_dict_generator(
        'Lab_L_value', L_list, need_differ=True)
    Lab_value_dict.update(_base_key_dict_generator(
        'Lab_a_value', a_list, need_differ=True))
    Lab_value_dict.update(_base_key_dict_generator(
        'Lab_b_value', b_list, need_differ=True))
    return Lab_value_dict


def _extract_rgb_histogram(image_path):
    image = cv2.imread(image_path)
    b_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    g_histogram = cv2.calcHist([image], [1], None, [256], [0, 256])
    r_histogram = cv2.calcHist([image], [2], None, [256], [0, 256])
    
    return np.median(r_histogram.flatten()), np.median(g_histogram.flatten()), np.median(b_histogram.flatten())


def rgb_histogram(images_path_list):
    r_list = []
    g_list = []
    b_list = []
    for image_path in images_path_list:
        r, g, b = _extract_rgb_histogram(image_path)
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
    rgb_histogram_dict = _base_key_dict_generator(
        'rgb_r_histogram', r_list, need_differ=True)
    rgb_histogram_dict.update(_base_key_dict_generator(
        'rgb_g_histogram', g_list, need_differ=True))
    rgb_histogram_dict.update(_base_key_dict_generator(
        'rgb_b_histogram', b_list, need_differ=True))
    return rgb_histogram_dict


def _extract_hsv_histogram(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_histogram = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    s_histogram = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_histogram = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    return np.median(h_histogram.flatten()), np.median(s_histogram.flatten()), np.median(v_histogram.flatten())


def hsv_histogram(images_path_list):
    h_list = []
    s_list = []
    v_list = []
    for image_path in images_path_list:
        h, s, v = _extract_hsv_histogram(image_path)
        h_list.append(h)
        s_list.append(s)
        v_list.append(v)
    hsv_histogram_dict = _base_key_dict_generator(
        'hsv_h_histogram', h_list, need_differ=True)
    hsv_histogram_dict.update(_base_key_dict_generator(
        'hsv_s_histogram', s_list, need_differ=True))
    hsv_histogram_dict.update(_base_key_dict_generator(
        'hsv_v_histogram', v_list, need_differ=True))
    return hsv_histogram_dict


def deal_all_clips_output(images_folder, output_table, lower=1, form_type='csv'):
    """處理一個資料夾下所有clip的切圖，讀取特征值，輸出csv
    """
    # types = ['H', 'A', 'SA', 'SU', 'D', 'F']
    result_dict = {}
    result_dict['emotion_type'] = []
    clips = []
    files = os.listdir(images_folder)
    for name in files:
        path = os.path.join(images_folder, name)
        if os.path.isdir(path):
            path_list = []
            current_images = os.listdir(path)
            if len(current_images) <= lower:
                continue
            for image in current_images:
                path_list.append(os.path.join(path, image))
            try:
                rgb_value_dict = rgb_value(path_list)
                for key, value in rgb_value_dict.items():
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(value)

                rgb_differ_dict = rgb_differ(path_list)
                for key, value in rgb_differ_dict.items():
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(value)

                hsv_value_dict = hsv_value(path_list)
                for key, value in hsv_value_dict.items():
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(value)

                hsl_value_dict = hsl_value(path_list)
                for key, value in hsl_value_dict.items():
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(value)

                Lab_value_dict = Lab_value(path_list)
                for key, value in Lab_value_dict.items():
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(value)

                rgb_histogram_dict = rgb_histogram(path_list)
                for key, value in rgb_histogram_dict.items():
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(value)

                hsv_histogram_dict = hsv_histogram(path_list)
                for key, value in hsv_histogram_dict.items():
                    if key not in result_dict:
                        result_dict[key] = []
                    result_dict[key].append(value)
            except OSError:
                print('OSError: cannot identify image file', name)
            except Exception as identifier:
                print(identifier, name)
            else:
                clips.append(name)
                if 'H' in name:
                    result_dict['emotion_type'].append('happy')
                elif 'SA' in name:
                    result_dict['emotion_type'].append('sad')
                elif 'SU' in name:
                    result_dict['emotion_type'].append('suprise')
                elif 'A' in name:
                    result_dict['emotion_type'].append('angry')
                elif 'D' in name:
                    result_dict['emotion_type'].append('disgust')
                elif 'F' in name:
                    result_dict['emotion_type'].append('fear')
            
    df = pd.DataFrame(result_dict, index=clips)
    if form_type == 'csv':
        df.to_csv(output_table, encoding='utf-8')
    elif form_type == 'xlsx':
         df.to_excel(output_table, encoding='utf-8')


def read_form(form_path, form_type='csv'):
    if form_type == 'csv':
        return pd.read_csv(form_path)
    elif form_type == 'xlsx':
        return pd.read_excel(form_path)


def z_score_df_column(dataframe):
    for col in dataframe.columns[1:]:  # 0: "Unnamed: 0"
        if col == 'emotion_type':
            continue
        else:
            dataframe[col] = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std(ddof=0)
    return dataframe


def plot_form_boxplot(dataframe, output_folder, normal='origianl'):
    """把每種數值按照各類型畫成boxplot
    """
    if output_folder[-1] != '/':
        output_folder += '/'

    if normal == 'z-score':
        dataframe = z_score_df_column(dataframe)

    # Divide to each emotional type df.
    emotion_df_dict = {
        'happy': dataframe[dataframe['emotion_type'] == 'happy'],
        'sad': dataframe[dataframe['emotion_type'] == 'sad'],
        'angry': dataframe[dataframe['emotion_type'] == 'angry'],
        'fear': dataframe[dataframe['emotion_type'] == 'fear'],
        'disgust': dataframe[dataframe['emotion_type'] == 'disgust'],
        'surprise': dataframe[dataframe['emotion_type'] == 'surprise']
    }

    # Boxplot of each tag with 6 emotions.
    for col in dataframe.columns[1:]:  # 0: "Unnamed: 0"
        if col == 'emotion_type':
            continue
        else:
            # different length boxplot sol.: https://blog.csdn.net/LeizRo/article/details/78524238
            emotion_col_serial_dict = {}
            for emotion, df in emotion_df_dict.items():
                # Do z-socre.
                emotion_col_serial_dict[emotion] = pd.Series(np.array(list(df[col])))

            current_pf_boxplot = pd.DataFrame(emotion_col_serial_dict)
            current_pf_boxplot.boxplot()
            plt.title(col + ' (' + normal + ')')
            plt.savefig(output_folder + col + '_' + normal + '.png')
            plt.clf()


def count_barplot(dataframe, output_path=None, base='emotion_type'):
    """畫出統計clip各類型的樣本數量
    """
    plt.figure()
    sns.countplot(x=base, data=dataframe)

    plt.title('Number of Videos')
    plt.xticks(rotation='vertical')
    plt.xlabel('emotion type')
    plt.ylabel('amount')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == '__main__':
    # # 【5 seconds to a clip ver.】
    # # Extract and write out.
    # clip_cut_images_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/clip_cut_images_folder'
    # output_table = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/video_feature.csv'
    # start = time.clock()
    # print('Start time:', datetime.datetime.now())
    # deal_all_clips_output(clip_cut_images_folder, output_table, lower=3, form_type='csv')
    # print('Stop time:', datetime.datetime.now())
    # elapsed = (time.clock() - start)
    # print("Time used:", elapsed)

    # # Boxplot.
    # boxplot_output_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/form_each_col_boxplot'
    # counting_boxplot_output_path = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/number_of_video.png'
    # form_path = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/video_feature.csv'
    # form_type = 'csv'
    # normal = 'z-score'
    # start = time.clock()
    # print('Start time:', datetime.datetime.now())
    # dataframe = read_form(form_path, form_type=form_type)
    # plot_form_boxplot(dataframe, boxplot_output_folder, normal=normal)
    # count_barplot(
    #     dataframe, output_path=counting_boxplot_output_path, base='emotion_type')
    # print('Stop time:', datetime.datetime.now())
    # elapsed = (time.clock() - start)
    # print("Time used:", elapsed)

    print('======')

    # 【5 min to a clip ver.】
    # Extract and write out.
    clip_cut_images_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/5min_clip_cut_images_folder'
    output_table = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/5min_video_feature.csv'
    start = time.clock()
    print('Start time:', datetime.datetime.now())
    deal_all_clips_output(clip_cut_images_folder,
                          output_table, lower=3, form_type='csv')
    print('Stop time:', datetime.datetime.now())
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)

    # Boxplot.
    boxplot_output_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/5_min_form_each_col_boxplot'
    counting_boxplot_output_path = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/5min_number_of_video.png'
    form_path = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/5min_video_feature.csv'
    form_type = 'csv'
    normal = 'z-score'
    start = time.clock()
    print('Start time:', datetime.datetime.now())
    dataframe = read_form(form_path, form_type=form_type)
    plot_form_boxplot(dataframe, boxplot_output_folder, normal=normal)
    count_barplot(
        dataframe, output_path=counting_boxplot_output_path, base='emotion_type')
    print('Stop time:', datetime.datetime.now())
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
