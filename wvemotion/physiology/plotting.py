#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Draw chart.
"""
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os


def _add_labels(rectangles):
    """Add label on each rectangle of a histogram.
    """
    for rectangle in rectangles:
        height = rectangle.get_height()
        x = rectangle.get_x() + rectangle.get_width() / 2
        y = height
        plt.text(x, y, height, ha='center', va='bottom')
        rectangle.set_edgecolor('white')


def plot_multilayer_lind_chart(x_axis_list, y_axis_dict, x_name,
                               save_path_pre, color_list, title,
                               fig_size=(8, 6), font_size=10, show=False):
    # Correct x_axis_list to be narray.
    x_axis_list = np.array(x_axis_list)
    # Set base of the chart
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['figure.figsize'] = fig_size

    lines = []
    kind_list = []
    times = 0
    # Generate lines.
    for kind in y_axis_dict.keys():
        kind_list.append(kind)
        lines.append(
            plt.plot(x_axis_list, y_axis_dict[kind], label=kind,
                     color=color_list[times], marker='o', linewidth=2))
        times += 1
    plt.xticks(x_axis_list, x_name)
    plt.title(title)
    plt.legend()
    save_path = save_path_pre
    if save_path[-1] != '/':
        save_path += '/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    save_path += title + '.png'
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_multilayer_histogram(x_axis_list, y_axis_dict, x_name,
                              save_path_pre, color_list, title,
                              fig_size=(8, 6), font_size=10,
                              bar_width=0.3, labels=False, show=False):
    """Plot multilayer histogram.
    """
    # Correct x_axis_list to be narray.
    x_axis_list = np.array(x_axis_list)
    # Set base of the chart
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['figure.figsize'] = fig_size

    rectangles = []
    kind_list = []
    times = 0
    # Generate rectangles.
    for kind in y_axis_dict.keys():
        kind_list.append(kind)
        rectangles.append(
            plt.bar(x_axis_list + bar_width * times, y_axis_dict[kind],
                    bar_width, color=color_list[times], label=kind))
        if labels:
            _add_labels(rectangles[times])
        times += 1
    plt.xticks(x_axis_list + bar_width, x_name)
    plt.title(title)
    plt.legend(handles=rectangles, labels=kind_list,  loc='best')
    save_path = save_path_pre
    if save_path[-1] != '/':
        save_path += '/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    save_path += title + '.png'
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # # test plot_multilayer_histogram
    # test = {'v1': {'happy': [1, 1, 1],
    #                             'fear': [2, 2, 2],
    #                             'sad': [3, 3, 3]},
    #         'v2': {'happy': [1.1, 1.1, 1.1, 1.1],
    #                'fear': [2.2, 2.2, 2.2, 2.2],
    #                'sad': [3.3, 3.3, 3.3, 3.3]}}
    # label_amount = [3, 4]
    # index = 0
    # for v in test.keys():
    #     x_axis_list = [index for index in range(1, label_amount[index] + 1)]
    #     plot_multilayer_histogram(x_axis_list, test[v], x_axis_list, 'test', ['yellow', 'red', 'blue'], 'test', 0.3, False, False)
    #     index += 1
    pass
