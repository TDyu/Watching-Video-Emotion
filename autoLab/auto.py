# -*- coding: utf-8 -*-
# need admin.
from pywinauto.application import Application  # need in the top (do not why...)
import configparser
import logging
import os
import random
import subprocess
import time
import tkinter as tk
import tkinter.font
import win32api

import filetype
from moviepy.video.io.VideoFileClip import VideoFileClip
import pyautogui

from experiment_logging import BASELINE_INFO
from experiment_logging import BREAK_INFO
from experiment_logging import COUNTDOWN_INFO
from experiment_logging import PLAY_INFO
from experiment_logging import RECORD_INFO
from experiment_logging import MEASURE_INFO
from experiment_logging import Q_INFO
from experiment_logging import ANSWER


class Video(object):
    """
    Attributes:
        name: str, Video name.
        path: str, Video path.
        seconds = int, Seconds of the video.
    """

    def __init__(self, name, path):
        """Create a new Video object.

        Args:
            name: str, Video name.
            path: str, Video path.
        """
        super(Video, self).__init__()
        self.name = name
        self.path = path
        self.seconds = self.__set_second_length()
        self.formatted_time = self.__set_formatted_time()

    def __set_second_length(self):
        """Get the seconds of the video.

        Returns:
            A int representation of seconds of the video.
        """
        clip = VideoFileClip(self.path)
        duration = clip.duration
        second = int(duration)
        return second

    def __set_formatted_time(self):
        """Transe seconds to formatted time. %M:%S

        Returns:
            A str representation of formatted time of video.
        """
        struct_time = time.localtime(self.seconds)
        return time.strftime("%M:%S", struct_time)

    def __str__(self):
        return self.name


class Window(tk.Tk):
    """
    Note:
        http://www.itguai.com/python/a4658757.html
    """

    def __init__(self, first_seconds, first_notice_message,
        notice_seconds, note_test):
        """Create a new Timer.
        Args:
            first_seconds: int, First stage seconds.
            first_notice_message: str, Notice message for first stage.
            notice_seconds: int, Countdown seconds.
            note_test: str, Notice message.
        """
        tk.Tk.__init__(self)
        # https://blog.csdn.net/asdf54sdf/article/details/50495942
        self.attributes('-fullscreen', True)
        # Set notice message label.
        if first_notice_message:
            self.notice_label = tk.Label(
                self, text=first_notice_message, font=('Helvetica', 100))
        else:
            self.notice_label = tk.Label(
                self, text='', font=('Helvetica', 100))
        self.notice_label.pack()
        # Set timer label.
        self.time_label = tk.Label(self, text='', font=('Helvetica', 150))
        self.time_label.pack()
        self.remaining = 0
        self.is_first_stage = True
        self.notice_seconds = notice_seconds
        self.first_notice_message = first_notice_message
        self.notice_message = note_test
        self.countdown(first_seconds)

    def countdown(self, remaining=None):
        if remaining is not None:
            self.remaining = remaining

        if self.remaining <= 0:
            if self.is_first_stage:
                self.is_first_stage = False
                COUNTDOWN_INFO()
                self.countdown(self.notice_seconds)
            else:
                COUNTDOWN_INFO(False)
                self.destroy()
        else:
            if not self.is_first_stage:
                self.notice_label.configure(text=self.notice_message)
                self.time_label.configure(text='%d' % self.remaining)
            self.remaining = self.remaining - 1
            self.after(1000, self.countdown)


class QWindow(tk.Tk):

    def __init__(self, seconds, video_name):
        tk.Tk.__init__(self)
        self.attributes('-fullscreen', True)
        self.grid()

        self.emotion_photos()
        self.notice_text()

        self.remaining = 0
        self.countdown(seconds)

        self.key_event_bind()

        self.set_display_text()
        self.video_name = video_name
        self.answers = {
            'A': 0,
            'S': 0,
            'D': 0,
            'F': 0,
            'G': 0,
            'H': 0}

    def countdown(self, remaining=None):
        if remaining is not None:
            self.remaining = remaining

        if self.remaining <= 0:
            print(str(self.answers))
            ANSWER(self.video_name, self.answers)
            self.destroy()
        else:
            self.label.configure(text="%d" % self.remaining)
            self.remaining = self.remaining - 1
            self.after(1000, self.countdown)

    def notice_text(self):
        ft = tkinter.font.Font(family='Fixdsys', size=40,
                               weight=tk.font.BOLD)

        tk.Label(text='請選擇觀片後的情緒，可複選', font=ft).grid(
            row=0, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='(Please select the emotions after viewing the film)',
                 font=ft).grid(row=2, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='步驟1 請先按字母選擇情緒', font=ft).grid(
            row=4, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='(Step 1: Please press the letters)', font=ft).grid(
            row=6, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='步驟2 再請選擇情緒程度0~5(低~高)，請按數字', font=ft).grid(
            row=203, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='Step 2 (Please choose the number 0 to 5)', font=ft).grid(
            row=206, column=3, columnspan=6, sticky=tk.W + tk.E)

    def emotion_photos(self):
        ft = tkinter.font.Font(family='Fixdsys', size=25, weight=tk.font.BOLD)

        photo_happy = tk.PhotoImage(file="./emotion/happy.png")
        self.label = tk.Label(image=photo_happy)
        self.label.image = photo_happy
        self.label.grid(row=200, column=2)
        tk.Label(text='A', font=ft).grid(row=201, column=2,
                                         sticky=tk.W + tk.E + tk.N + tk.S)

        photo_surprise = tk.PhotoImage(file="./emotion/surprise.png")
        self.label = tk.Label(image=photo_surprise)
        self.label.image = photo_surprise
        self.label.grid(row=200, column=3)
        tk.Label(text='S', font=ft).grid(
            row=201, column=3, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_afraid = tk.PhotoImage(file="./emotion/afraid.png")
        self.label = tk.Label(image=photo_afraid)
        self.label.image = photo_afraid
        self.label.grid(row=200, column=4)
        tk.Label(text='D', font=ft).grid(
            row=201, column=4, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_sad = tk.PhotoImage(
            file="./emotion/sad.png")
        self.label = tk.Label(image=photo_sad)
        self.label.image = photo_sad
        self.label.grid(row=200, column=5)
        tk.Label(text='F', font=ft).grid(
            row=201, column=5, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_angry = tk.PhotoImage(
            file="./emotion/angry.png")
        self.label = tk.Label(image=photo_angry)
        self.label.image = photo_angry
        self.label.grid(row=200, column=6)
        tk.Label(text='G', font=ft).grid(
            row=201, column=6, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_disgust = tk.PhotoImage(
            file="./emotion/disgust.png")
        self.label = tk.Label(image=photo_disgust)
        self.label.image = photo_disgust
        self.label.grid(row=200, column=7)
        tk.Label(text='H', font=ft).grid(
            row=201, column=7, sticky=tk.W + tk.E + tk.N + tk.S)

    def set_display_text(self):
        ft = tkinter.font.Font(family='Fixdsys', size=20,
                               weight=tk.font.BOLD)
        self.display_label_text = tk.StringVar()
        self.display_label_text.set('已選擇:')
        self.display_label = tk.Label(textvariable=self.display_label_text, font=ft, bg='white').grid(
            row=210, column=5, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)

    def A_Key(self, event):
        if not self.has_A:
            self.has_choice = True
            text = self.display_label_text.get() + '\nA   '
            self.display_label_text.set(text)
            self.has_A = True
            self.press = 'A'

    def S_Key(self, event):
        if not self.has_S:
            self.has_choice = True
            text = self.display_label_text.get() + '\nS   '
            self.display_label_text.set(text)
            self.has_S = True
            self.press = 'S'

    def D_Key(self, event):
        if not self.has_D:
            self.has_choice = True
            text = self.display_label_text.get() + '\nD   '
            self.display_label_text.set(text)
            self.has_D = True
            self.press = 'D'

    def F_Key(self, event):
        if not self.has_F:
            self.has_choice = True
            text = self.display_label_text.get() + '\nF   '
            self.display_label_text.set(text)
            self.has_F = True
            self.press = 'F'

    def G_Key(self, event):
        if not self.has_G:
            self.has_choice = True
            text = self.display_label_text.get() + '\nG   '
            self.display_label_text.set(text)
            self.has_G = True
            self.press = 'G'

    def H_Key(self, event):
        if not self.has_H:
            self.has_choice = True
            text = self.display_label_text.get() + '\nH   '
            self.display_label_text.set(text)
            self.has_H = True
            self.press = 'H'

    def One_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   1'
            self.display_label_text.set(text)
            self.answers[self.press] = 1
            self.has_choice = False
            self.press = None

    def Two_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   2'
            self.display_label_text.set(text)
            self.answers[self.press] = 2
            self.has_choice = False
            self.press = None

    def Three_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   3'
            self.display_label_text.set(text)
            self.answers[self.press] = 3
            self.has_choice = False
            self.press = None

    def Four_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   4'
            self.display_label_text.set(text)
            self.answers[self.press] = 4
            self.has_choice = False
            self.press = None

    def Five_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   5'
            self.display_label_text.set(text)
            self.answers[self.press] = 5
            self.has_choice = False
            self.press = None

    def key_event_bind(self):
        self.has_A = False
        self.has_S = False
        self.has_D = False
        self.has_F = False
        self.has_G = False
        self.has_H = False
        self.has_choice = False
        self.press = None

        self.bind('a', self.A_Key)
        self.bind('s', self.S_Key)
        self.bind('d', self.D_Key)
        self.bind('f', self.F_Key)
        self.bind('g', self.G_Key)
        self.bind('h', self.H_Key)

        self.bind('1', self.One_Key)
        self.bind('2', self.Two_Key)
        self.bind('3', self.Three_Key)
        self.bind('4', self.Four_Key)
        self.bind('5', self.Five_Key)


def configurate():
    """Configurate path.
    Returns:
        A dict representation of video_folder_path and player_path.
    """
    config_parser = configparser.ConfigParser()
    # Keep capitalization of the original letter.
    config_parser.optionxform = str
    base_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(base_dir, 'experiment.config')
    config = {
        'video_folder_path': None,
        'player_path': None,
        'labchart_path': None,
        'sampling_img_path': None,
        'camera_path': None,
        'base_line': None,
        'break': None,
        'notice_countdown': None,
        'q_countdown': None,
        'video_types': None
    }
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        config_parser.readfp(config_file)
        config['video_folder_path'] = config_parser.get(
            'path', 'video_folder_path')
        config['player_path'] = config_parser.get(
            'path', 'player_path')
        config['labchart_path'] = config_parser.get(
            'path', 'labchart_path')
        config['sampling_img_path'] = config_parser.get(
            'path', 'sampling_img_path')
        config['camera_path'] = config_parser.get(
            'path', 'camera_path')
        config['base_line'] = int(config_parser.get(
            'time', 'base_line'))
        config['break'] = int(config_parser.get(
            'time', 'break'))
        config['notice_countdown'] = int(config_parser.get(
            'time', 'notice_countdown'))
        config['q_countdown'] = int(config_parser.get(
            'time', 'q_countdown'))
        values = config_parser.get(
            'video_types', 'types')
        config['video_types'] = list(map(str, values.split(',')))
    return config


def get_video_objects_list(folder_path, video_types):
    """Get all video objects in the folder.

    Args:
        folder_path: str, Path of taget videos folder.
        video_types: list of str, Types of testing videos.

    Returns:
        A list of Video representation of videos.
    """
    video_objects_list = []
    absolute_folder_path = os.path.abspath(folder_path)
    files = os.listdir(absolute_folder_path)
    for file_name in files:
        file_path = absolute_folder_path
        if folder_path[-1] != '/':
            file_path += '/' + file_name
        else:
            file_path += file_name
        file_path = os.path.abspath(file_path)
        # Check the file is video type.
        # If it is not video type, the function will return None.
        # Or, it will return type instance.
        # https://h2non.github.io/filetype.py/v0.1.5/match.m.html#filetype.match.video_matchers
        # if filetype.video(file_path):
        #     video_objects_list.append(Video(file_name, file_path))
        if file_name[-3:] in video_types:
            video_objects_list.append(Video(file_name, file_path))
    return video_objects_list


def click_measuring(labchart_path, sampling_img_path):
    app = Application().connect(path=labchart_path)
    labchart = app.LabChart
    labchart.Wait('ready')
    labchart.Maximize()
    # Sure the window open.
    time.sleep(1)
    sampling_button_location = pyautogui.locateOnScreen(
        sampling_img_path)
    left, top, width, height = sampling_button_location
    center_x, center_y = pyautogui.center((left, top, width, height))
    pyautogui.moveTo(center_x, center_y - 3)
    pyautogui.click()


def click_record(camera_path, is_start=True):
    app = Application().connect(path=camera_path)
    camera = app.Dialog
    camera.Wait('ready')
    if is_start:
        camera.TypeKeys('{F5}')
    else:
        camera.TypeKeys('{F7}')


def control_player(player_path):
    """Control player to be full screen playing after command play a video.

    Args:
        player_path: str, Path of player exe.

    Returns:
        A object representation of the player app.
    """
    app = Application().connect(path=player_path)
    potplayer = app.Potplayer64
    potplayer.Wait('ready')
    potplayer.type_keys('{ENTER}')
    return potplayer


def create_countdown_window(first_seconds, first_notice_message,
                            notice_seconds, notice_message):
    window = Window(first_seconds, first_notice_message,
                    notice_seconds, notice_message)
    window.mainloop()


def write_questionnaire(seconds, video_name):
     window = QWindow(seconds, video_name)
     window.mainloop()


def open_player(video_folder_path, video_file, video_seconds):
    if video_folder_path[-1] != '/':
        video_folder_path += '/'
    video_path = video_folder_path + str(video_file)
    win32api.ShellExecute(0, 'open', video_path, '', '', 1)
    time.sleep(video_seconds + 5)


def open_player_sub(player_path, video_path):
    player = subprocess.Popen([player_path, video_path], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    player.communicate()

if __name__ == '__main__':
    # Get path and timing setting from cofig.
    config = configurate()
    video_folder_path = str(config['video_folder_path'])
    player_path = config['player_path']
    labchart_path = config['labchart_path']
    sampling_img_path = config['sampling_img_path']
    camera_path = config['camera_path']
    base_line_seconds = config['base_line']
    break_seconds = config['break']
    notice_countdown_seconds = config['notice_countdown']
    q_countdown = config['q_countdown']
    video_types = config['video_types']
    baseline_notice_message = None
    break_notice_message = '~休息時間~\n(可以亂動但請勿拔掉儀器)'
    notice_countdown_message = '即將開始觀看影片'

    # Get videos and shuffle.
    videos = get_video_objects_list(video_folder_path, video_types)
    random.shuffle(videos)


    for video in videos:
        print(video)
        # commandline = '"' + player_path + '" ' + '"' + video.path + '"'
        # subprocess.call(commandline)
        # open_player(video_folder_path, video.name, video.seconds)
        open_player_sub(player_path, video.path)
        write_questionnaire(q_countdown, video.name)

    # Start to record
    click_record(camera_path)
    RECORD_INFO()

    # Start measuring.
    click_measuring(labchart_path, sampling_img_path)
    MEASURE_INFO()

    # base line time
    BASELINE_INFO()
    create_countdown_window(
        base_line_seconds - notice_countdown_seconds,
        baseline_notice_message,
        notice_countdown_seconds,
        notice_countdown_message)
    BASELINE_INFO(False)

    # Start to do cycle process of experiment.
    count = 0
    for video in videos:
        count += 1
        # Play video.
        PLAY_INFO(video.name, video.formatted_time, video.seconds)
        open_player_sub(player_path, video.path)
        PLAY_INFO(video.name, video.formatted_time, video.seconds, False)
        # Do questionnaire.
        Q_INFO()
        write_questionnaire(q_countdown, video.name)
        Q_INFO(False)
        # Take break if video is not last.
        if count != len(videos):
            BREAK_INFO()
            create_countdown_window(
                break_seconds - notice_countdown_seconds,
                break_notice_message,
                notice_countdown_seconds,
                notice_countdown_message)
            BREAK_INFO(False)

    # Stop measuring.
    click_measuring(labchart_path, sampling_img_path)
    MEASURE_INFO(False)

    # Stop record
    RECORD_INFO(False)
    click_record(camera_path, False)
