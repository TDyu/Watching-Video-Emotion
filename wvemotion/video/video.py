#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Operation about videos.
"""
import re
import os
import shutil

from cv2 import cv2
from moviepy.editor import VideoFileClip
from PIL import Image


class VideoFile(object):
    """包裝視頻訊息
    """

    def __init__(self, full_name, path):
        """Create a object for VideoFile.

        Args:
            full_name: str, The video name with file extension.
            path: str, Tha abosolute path or relative path of the video file.
        """
        super(VideoFile, self).__init__()
        self.name = ''
        self.type = ''
        self.abspath = os.path.abspath(path)
        # self.seconds = self.get_file_seconds()
        self.fps = self.get_fps()
        self.__set_name_type(full_name)

    def __set_name_type(self, full_name):
        """Divide name and file extension.
        """
        dot_index = full_name.rfind('.')
        self.name = full_name[:dot_index]
        self.type = full_name[dot_index+1:]

    def get_fps(self):
        """取得fps
        """
        fps = 25.0
        videoCapture = cv2.VideoCapture(self.abspath)
        # Check open.
        if videoCapture.isOpened():
            is_open = videoCapture.read()[0]
        else:
            is_open = False
        # Get fps.
        if is_open:
            # https://www.jianshu.com/p/c805b4803755
            fps =  videoCapture.get(cv2.CAP_PROP_FPS)
        videoCapture.release()
        return fps
    
    def get_file_seconds(self):
        """取得時間長（s）
        https://blog.csdn.net/xiaomahuan/article/details/78783174
        """
        clip = VideoFileClip(self.abspath)
        duration = clip.duration
        # Remember to close, or it will OSError 控制代码无效.
        # https://stackoverflow.com/questions/43966523/getting-oserror-winerror-6-the-handle-is-invalid-in-videofileclip-function
        clip.reader.close()
        clip.audio.reader.close_proc()
        clip.close()
        return duration

    def get_filetime_str(self):
        minute = 60
        hour = 60 ** 2
        seconds = self.get_file_seconds()
        if seconds < minute:
            return str(seconds) + 's'
        if seconds < hour:
            return '%sm %ss' % (int(seconds / minute), int(seconds % minute))
        else:
            video_hour = int(seconds / hour)
            video_minute = int(seconds % hour / minute)
            video_second = int(seconds % hour % minute)
            return '%sh %sm %ss' % (video_hour, video_minute, video_second)

    def get_filebyte(self):
        """Get file bytes.
        """
        return os.path.getsize(self.abspath)

    def get_filesize_str(self):
        """Get formated file string.
        """
        bytes = self.get_filebyte()
        kb = 1024
        mb = 1024 ** 2
        gb = 1024 ** 3
        if bytes >= gb:
            return str(bytes / gb)+'GB'
        elif bytes >= mb:
            return str(bytes / mb)+'MB'
        elif bytes >= kb:
            return str(bytes / kb)+'KB'
        else:
            return str(bytes)+'Bytes'

    def __check_cut_time(self, start_time, end_time=0, duration=0, is_duration=False):
        seconds = self.get_file_seconds()
        if is_duration:
            target = start_time + duration
            if target > seconds:
                return {start_time: seconds}, True
            else:
                return {start_time: target}, False
        else:
            if end_time > seconds:
                return {start_time: seconds}, True
            else:
                return {start_time: end_time}, False

    def __calculate_duration_time(self, start_time, duration):
        corresponding_time_dict, is_over = self.__check_cut_time(start_time, duration=duration, is_duration=True)
        while not is_over:
            # start_time = last end time
            start_time = corresponding_time_dict[start_time]
            corresponding, is_over = self.__check_cut_time(start_time, duration=duration, is_duration=True)
            corresponding_time_dict.update(corresponding)
        return corresponding_time_dict

    def __cut_video_by_time(self, output_path, start_time, end_time):
        """
        https://gist.github.com/dslwind/93bc63546524d81c26ebd846251b00a1
        https://blog.csdn.net/dszgf5717/article/details/80868724
        """
        format_dict = {
            'start_second': start_time,
            'end_second': end_time,
            'input_path': self.abspath,
            'output_path': output_path
        }
        command_format = 'ffmpeg -ss %(start_second)d -to %(end_second)d -i "%(input_path)s" -c:v h264 -c:a copy "%(output_path)s"' % format_dict
        os.system(command_format)

    def __check_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def cut_video(self, output_folder='./', output_name='', start_time=0, end_time=0, duration=0, is_duration=False, is_each=False):
        """根據時間切割視頻
        """
        self.__check_folder(output_folder)
        output_path = output_folder
        if output_path[-1] != '/':
            output_path += '/'
        if output_name == '':
            output_name = self.name
        output_path += output_name + '_cut'

        if is_duration and is_each:
            corresponding_time_dict = self.__calculate_duration_time(start_time, duration)
            count = 1
            output_path += '_'
            for start_time, end_time in corresponding_time_dict.items():
                current_path = output_path + str(count) + '.' + self.type
                self.__cut_video_by_time(current_path, start_time, end_time)
                count += 1
        elif is_duration:
            corresponding_time_dict = self.__check_cut_time(start_time, duration=duration, is_duration=True)[0]
            output_path += '.' + self.type
            self.__cut_video_by_time(output_path, start_time, corresponding_time_dict[start_time])
        else:
            corresponding_time_dict = self.__check_cut_time(start_time, end_time=end_time)[0]
            output_path += '.' + self.type
            self.__cut_video_by_time(output_path, start_time, corresponding_time_dict[start_time])

    def cut_images(self, output_folder='./', interval=1):
        """
        https://blog.csdn.net/xinxing__8185/article/details/48440133
        """
        self.__check_folder(output_folder)
        if output_folder[-1] != '/':
            output_folder += '/'
        output_path = output_folder + self.name + '_'

        videoCapture = cv2.VideoCapture(self.abspath)
        # Check open.
        if videoCapture.isOpened():
            is_open, frame = videoCapture.read()
        else:
            is_open = False

        frame_interval = int(self.fps * interval)

        count = 1
        image_interval = interval
        while is_open:
            is_open, frame = videoCapture.read()
            if(count % frame_interval == 0):
                image_path = output_path + str(image_interval) + '.jpg'
                # * The path need to be all english.
                cv2.imwrite(image_path, frame)
                image_interval += interval
                # crop_image = self.__crop_black(image_path)
                # cv2.imwrite(image_path, crop_image)
            count += 1
            cv2.waitKey(1)
        videoCapture.release()
        return output_folder

    def __is_crust(self, pix):
        return sum(pix) < 25


    def __high_check(self, img, y, step=50):
        count = 0
        width = img.size[0]
        for x in range(0, width, step):
            if self.__is_crust(img.getpixel((x, y))):
                count += 1
            if count > width / step / 2:
                return True
        return False

    def __wide_check(self, img, x, step=50):
        count = 0
        height = img.size[1]
        for y in range(0, height, step):
            if self.__is_crust(img.getpixel((x, y))):
                count += 1
            if count > height / step / 2:
                return True
        return False

    def __find_boundary(self, img, crust_side, core_side, checker):
        if not checker(img, crust_side):
            return crust_side
        if checker(img, core_side):
            return core_side

        mid = (crust_side + core_side) / 2
        while mid != core_side and mid != crust_side:
            if checker(img, mid):
                crust_side = mid
            else:
                core_side = mid
            mid = (crust_side + core_side) / 2
        return core_side

    def __handle_image_judge_zise(self, original_images_path, rect_dict):
        img = Image.open(original_images_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        width, height = img.size

        left = self.__find_boundary(img, 0, width / 2, self.__wide_check)
        right = self.__find_boundary(img, width - 1, width / 2, self.__wide_check)
        top = self.__find_boundary(img, 0, height / 2, self.__high_check)
        bottom = self.__find_boundary(img, height - 1, width / 2, self.__high_check)

        rect = (left, top, right, bottom)
        # print(rect)
        if rect not in rect_dict:
            rect_dict[rect] = 1
        else:
            rect_dict[rect] += 1
        img.close()
        return rect_dict

    def __find_max_times_rect(self, rect_dict):
        max_times = 0
        max_rect = None

        for key, value in rect_dict.items():
            if value > max_times:
                max_times = value
                max_rect = key

        # print(max_rect)
        # print(max_times)

        return max_rect

    def __juge_size(self, original_images_folder):
        # First traversal: record the size of the black border of each image.
        rect_dict = {}
        for filename in os.listdir(original_images_folder):
            if filename.split('.')[-1].upper() in ('JPG', 'JPEG', 'PNG', 'BMP', 'GIF'):
                rect_dict = self.__handle_image_judge_zise(
                    os.path.join(original_images_folder, filename), rect_dict)
        # Find the most likely black edge size (in majority).
        return self.__find_max_times_rect(rect_dict)

    def __handle_image_cut(self, original_images_path, target_folder, size_tuple):
        img = Image.open(original_images_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        rect = size_tuple

        region = img.crop(rect)
        filename = os.path.split(original_images_path)[-1]
        try:
            region.save(os.path.join(target_folder, filename), 'PNG')
        except SystemError:
            # shutil.move(original_images_path,
            #             os.path.join(target_folder, filename))
            pass
        img.close()

    def scorp_black(self, original_images_folder):
        # Find the most likely black edge size (in majority).
        size_tuple = self.__juge_size(original_images_folder)
        # Second traversal: scrop the black border.
        # target_folder = original_images_folder + 'scrop_black/'
        target_folder = original_images_folder
        for filename in os.listdir(original_images_folder):
            if filename.split('.')[-1].upper() in ('JPG', 'JPEG', 'PNG', 'BMP', 'GIF'):
                self.__handle_image_cut(os.path.join(
                    original_images_folder, filename), target_folder, size_tuple)


def get_all_videos(folder_path):
    """Get all videos in folder_path.
    Args:
        folder_path: str, The path of videos .

    Returns:
        A list of VideoFile for representation of videos.
    """
    videos_type = ['mp4', 'avi', 'flv', 'mpg', 'mkv']
    videos = []

    # Get all files and child folders in the folder_path.
    files = os.listdir(folder_path)
    for name in files:
        relative_path = os.path.join(folder_path, name)
        if os.path.isfile(relative_path):
            absolute_path = os.path.abspath(relative_path)
            file_extension = os.path.splitext(absolute_path)[-1][1:]
            if file_extension in videos_type:
                try:
                    videos.append(VideoFile(name, absolute_path))
                except OSError:
                    print(name, absolute_path)
    return videos


def get_all_clip_videos(folder_path):
    videos_type = ['mp4', 'avi', 'flv', 'mpg', 'mkv']
    videos = []

    files = os.listdir(folder_path)
    for name in files:
        relative_path = os.path.join(folder_path, name)
        absolute_path = os.path.abspath(relative_path)
        if os.path.isdir(absolute_path) and 'cut' in name:
            clips_files = os.listdir(absolute_path)
            for clip in clips_files:
                clip_path = os.path.join(absolute_path, clip)
                if os.path.isfile(clip_path):
                    file_extension = os.path.splitext(clip_path)[-1][1:]
                    if file_extension in videos_type:
                        videos.append(VideoFile(name, absolute_path))
    return videos


def cut_multiple_videos(videos, output_folder, start_time=0, end_time=0, duration=5, is_duration=False, is_each=False):
    if output_folder[-1] != '/':
        output_folder += '/'
    for video in videos:
        current_output_folder = output_folder + video.name + '_cut/'
        video.cut_video(output_folder=current_output_folder, start_time=start_time,
                        end_time=end_time, duration=duration, is_duration=is_duration, is_each=is_each)


def cut_multiple_videos_images(videos, output_folder):
    if output_folder[-1] != '/':
        output_folder += '/'
    for video in videos:
        current_output_folder = output_folder + video.name + '_cut_images/'
        current_output_folder = video.cut_images(
            output_folder=current_output_folder, interval=1)
        video.scorp_black(current_output_folder)


def deal_all_videos(folder_path, clips_output_folder, images_output_folder, start_time=0, end_time=0, duration=5, is_duration=True, is_each=True):
    if clips_output_folder[-1] != '/':
        clips_output_folder += '/'
    if images_output_folder[-1] != '/':
        images_output_folder += '/'

    videos_type = ['mp4', 'avi', 'flv', 'mpg', 'mkv']

    # Get all files and child folders in the folder_path.
    files = os.listdir(folder_path)
    for name in files:
        relative_path = os.path.join(folder_path, name)
        if os.path.isfile(relative_path):
            absolute_path = os.path.abspath(relative_path)
            file_extension = os.path.splitext(absolute_path)[-1][1:]
            if file_extension in videos_type:
                try:
                    # Original video.
                    video = VideoFile(name, absolute_path)

                    # Cut to videos.
                    current_clips_output_folder = clips_output_folder + video.name + '_cut/'
                    video.cut_video(output_folder=current_clips_output_folder, start_time=start_time,
                                    end_time=end_time, duration=duration, is_duration=is_duration, is_each=is_each)

                    # Cut images for each video clip.
                    for clip in os.listdir(current_clips_output_folder):
                        clip_abspath = os.path.join(
                            current_clips_output_folder, clip)
                        clip_file_extension = os.path.splitext(
                            clip_abspath)[-1][1:]
                        if clip_file_extension in videos_type:
                            clip = VideoFile(clip, clip_abspath)
                            current_clip_images_folder = images_output_folder + clip.name + '_cut_images/'
                            current_clip_images_folder = clip.cut_images(
                                output_folder=current_clip_images_folder, interval=1)
                            clip.scorp_black(current_clip_images_folder)
                except OSError:
                    print('OSError', name, absolute_path)


def cut_images_and_dispatch(video_full_name, video_path, output_folder, frame_interval=1, dispatch_unit=5):
    """處理一部影片的切圖、去黑邊，并把圖分配成一個clip一個資料夾
    frame_interval：幾秒切一張
    dispatch_unit：幾秒為一個clip
    """
    # Cut images.
    if output_folder[-1] != '/':
        output_folder += '/'
    video = VideoFile(
        video_full_name, video_path)
    output_folder = video.cut_images(output_folder, interval=frame_interval)
    video.scorp_black(output_folder)

    # Dispatch images.
    for image in os.listdir(output_folder):
        path = os.path.join(output_folder, image)
        if os.path.isfile(path):
            file_extension = os.path.splitext(path)[-1][1:]
            if file_extension.upper() in ('JPG', 'JPEG', 'PNG', 'BMP', 'GIF'):
                num = int(re.findall('[0-9]+', image)[1])
                to_class = (num - 1) // dispatch_unit + 1
                base_images_folder_name = output_folder + video.name + '_cut_%d_cut_images/'
                current_clip_images_folder = base_images_folder_name % to_class
                if not os.path.exists(current_clip_images_folder):
                        os.makedirs(current_clip_images_folder)
                shutil.move(path, current_clip_images_folder)


def deal_all_videos_cut_images_and_dispath(videos_folder, output_folder, frame_interval=1, dispatch_unit=5):
    """處理一個資料夾下所有影片，處理切圖、去黑邊，并把圖分配成一個clip一個資料夾
    frame_interval：幾秒切一張
    dispatch_unit：幾秒為一個clip
    """
    if output_folder[-1] != '/':
        output_folder += '/'

    videos_type = ['mp4', 'avi', 'flv', 'mpg', 'mkv']

    # Get all files and child folders in the folder_path.
    files = os.listdir(videos_folder)
    for name in files:
        relative_path = os.path.join(videos_folder, name)
        if os.path.isfile(relative_path):
            absolute_path = os.path.abspath(relative_path)
            file_extension = os.path.splitext(absolute_path)[-1][1:]
            if file_extension in videos_type:
                cut_images_and_dispatch(
                    name, absolute_path, output_folder, frame_interval=frame_interval, dispatch_unit=dispatch_unit)


if __name__ == '__main__':
    # # 【5 seconds to a clip ver.】
    # videos_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/all_videos'
    # # # videos = get_all_videos(videos_folder)
    # # # cut_multiple_videos(videos, videos_folder, duration=5,
    # # #                     is_duration=True, is_each=True)
    # clip_cut_images_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/clip_cut_images_folder'
    # # # clips_videos = get_all_clip_videos(videos_folder)
    # # # cut_multiple_videos_images(clips_videos, clip_cut_images_folder)
    # # deal_all_videos(videos_folder, videos_folder, clip_cut_images_folder,
    # #                 start_time=0, end_time=0, duration=5, is_duration=True, is_each=True)


    # # cut_images_and_dispatch('1-A.mp4', 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/all_videos/1-A.mp4',
    # #                         'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/clip_cut_images_folder', frame_interval=1, dispatch_unit=5)

    # deal_all_videos_cut_images_and_dispath(
    #     videos_folder, clip_cut_images_folder, frame_interval=1, dispatch_unit=5)

    print('======')

    # 【5 min to a clip ver.】
    videos_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/lab_videos'
    clip_cut_images_folder = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/5min_clip_cut_images_folder'
    deal_all_videos_cut_images_and_dispath(
        videos_folder, clip_cut_images_folder, frame_interval=1, dispatch_unit=5*60)
