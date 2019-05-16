#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Compare a video RMS similarity by each second frame.
"""
import copy
from functools import reduce
import math
import os
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def compare_rgb_rms(picture1, picture2):
    """Compare histogram RMS of rgb between two image.
    """
    image1 = Image.open(picture1)
    image2 = Image.open(picture2)

    histogram1 = image1.histogram()
    histogram2 = image2.histogram()

    differ = math.sqrt(reduce(operator.add, list(map(lambda a,b: (a-b) ** 2, histogram1, histogram2))) / len(histogram1))

    return differ


def get_video_similarity(image_folder_path):
    """Get a video RMS similarity by each second frame.

    Args:
        image_folder_path: str, The folder path has images of the viedeo.
    
    Return:
        A numpy.array is representation of the differ between each image.
    """
    # Get all files and folders in the folder.
    # Choose the files which are jpg.
    jpg_files = []
    files = os.listdir(image_folder_path)
    for f in files:
        relative_path = os.path.join(image_folder_path, f)
        if os.path.isfile(relative_path) and '.jpg' in f:
            jpg_files.append(relative_path)

    # Calculate the similarity.
    differs = []
    index = 1
    for jpg_file in jpg_files[1:]:
        differs.append(compare_rgb_rms(jpg_files[index - 1], jpg_file))
        index += 1

    return np.array(differs)


def get_multiple_videos_similarity(folder_path, need_same_shape=False):
    """Get multiple videos RMS similarity.

    Args:
        folder_path: str, The folder has some folders used to compare.

    Returns:
        A tuple of numpy.array or pandas.Serial(if need same shape) and a list is representation of differs of each
        video and compared name list.
    """
    # Get the compared target name and path.
    videos_name = []
    videos_images_path = []
    files = os.listdir(folder_path)
    for f in files:
        relative_path = os.path.join(folder_path, f)
        if os.path.isdir(relative_path):
            videos_name.append(f)
            videos_images_path.append(relative_path)

    # Get each video's similarity.
    multiple_video_differs = []
    for video_images_path in videos_images_path:
        multiple_video_differs.append(get_video_similarity(video_images_path))

    multiple_video_differs = np.array(multiple_video_differs)

    if need_same_shape:
        multiple_video_differs = pd.Series(multiple_video_differs)
    
    return multiple_video_differs, videos_name


def draw_boxplot(arrays, columns, title, target_path):
    index = 0
    if target_path[-1] != '/':
        target_path += '/'
    for array in arrays:
        df = pd.DataFrame(array, columns=[columns[index]])
        df.boxplot()
        plt.title(title)
        # plt.show()
        current_path = target_path + columns[index] + '.png'
        plt.savefig(current_path)
        plt.cla()
        index += 1


def draw_boxplot_one(arrays, columns, title, target_path):
    if target_path[-1] != '/':
        target_path += '/'
    # df = pd.DataFrame(arrays, columns=columns)
    data = pd.DataFrame(arrays)
    df = pd.DataFrame(columns=columns)
    df = df.append(data)
    df.boxplot()
    plt.title(title)
    # plt.show()
    current_path = target_path + 'dealed_same_shape.png'
    plt.savefig(current_path)
    plt.cla()


def draw_line_chart(arrays, columns, title, target_path):
    index = 0
    if target_path[-1] != '/':
        target_path += '/'
    for array in arrays:
        x_axis = np.array(list(range(1, len(array) + 1)))
        fig = plt.figure(figsize=(50, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(range(len(x_axis)))
        ax.set_xticklabels(x_axis, rotation=90)
        plt.plot(
            x_axis,
            array,
            linestyle='-',  # 折線型別
            linewidth=2,  # 折線寬度
            color='steelblue',  # 折線顏色
            marker='o',  # 點的形狀
            markersize=6,  # 點的大小
            markeredgecolor='black',  # 點的邊框色
            markerfacecolor='brown'  # 點的填充色
        )
        plt.title(title)
        plt.xlabel('frame')
        plt.ylabel('RMS')
        # 剔除圖框上邊界和右邊界的刻度
        plt.tick_params(top='off', right='off')
        # plt.show()
        current_path = target_path + columns[index] + '.png'
        plt.savefig(current_path)
        plt.cla()
        index += 1


if __name__ == '__main__':
    # # not same shape.
    # multiple_video_differs, videos_name = get_multiple_videos_similarity(
    #     'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/images/target')
    # # draw_boxplot(multiple_video_differs, videos_name,
    # #              'RGB-RMS', 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/boxplot_RGB_RMS')
    # draw_line_chart(multiple_video_differs, videos_name,
    #                 'RGB-RMS', 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/line_RGB_RMS')

    # deal to same shape.
    multiple_video_differs, videos_name = get_multiple_videos_similarity(
        'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/images/target', need_same_shape=False)
    draw_boxplot_one(multiple_video_differs,
                     videos_name, 'RGB-RMS', 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/chart_output/boxplot_RGB_RMS')
