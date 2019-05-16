#!/user/bin/env python3
# -*- coding: utf-8 -*-
# https://blog.csdn.net/CuGBabyBeaR/article/details/36213255
import os

import cv2


class Video(object):
    def __init__(self, name, abspath, file_extension):
        self.name = name
        self.abspath = abspath
        self.file_extension = file_extension
        self.fps = self.__set_fps()

    def __set_fps(self):
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
        else:
            fps = 25.0

        videoCapture.release()

        return fps

    def __crop_black(self, read_file):
        # https://blog.csdn.net/qq_38269799/article/details/80687830
        image = cv2.imread(read_file, 1)  # 读取图片 image_name应该是变量

        b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
        binary_image = b[1]  # 二值图--具有三通道
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        print(binary_image.shape)  # 改为单通道

        x = binary_image.shape[0]
        # print("高度x=", x)
        y = binary_image.shape[1]
        # print("宽度y=", y)
        edges_x = []
        edges_y = []

        for i in range(x):
            for j in range(y):
                if binary_image[i][j] == 255:
                    # print("横坐标",i)
                    # print("纵坐标",j)
                    edges_x.append(i)
                    edges_y.append(j)

        left = min(edges_x)  # 左边界
        right = max(edges_x)  # 右边界
        width = right - left  # 宽度

        bottom = min(edges_y)  # 底部
        top = max(edges_y)  # 顶部
        height = top - bottom  # 高度

        pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取

        return pre1_picture  # 返回图片数据

    def cut_images(self, parent_folder, seconds):
        # https://blog.csdn.net/xinxing__8185/article/details/48440133
        count = 1

        videoCapture = cv2.VideoCapture(self.abspath)
        # Check open.
        if videoCapture.isOpened():
            is_open, frame = videoCapture.read()
        else:
            is_open = False
        
        frame_interval = int(self.fps * seconds)
        
        while is_open:
            is_open, frame = videoCapture.read()
            if(count % frame_interval == 0):
                folder_path = self.__create_folder(parent_folder)
                image_path = folder_path + '/' + str(count) + '.jpg'
                cv2.imwrite(image_path, frame)
                # crop_image = self.__crop_black(image_path)
                # cv2.imwrite(image_path, crop_image)
            count += 1
            cv2.waitKey(1)
        videoCapture.release()

    def __create_folder(self, parent_folder):
        file_extension_index = self.name.rfind(self.file_extension)
        pure_name = self.name[:file_extension_index]
        folder_path = parent_folder

        if parent_folder[-1] != '/':
            folder_path += '/' + pure_name
        else:
            folder_path += pure_name

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        
        return folder_path

    def __str__(self):
        return self.name + '\n' + str(self.fps) + ' fps\n' + self.abspath + '\n' + self.file_extension


def get_all_videos(folder_path):
    """Get all videos in folder_path.
    Args:
        folder_path: str, The path of videos .

    Returns:
        A list of Video for representation of videos.
    """
    videos_type = ['mp4', 'avi', 'flv', 'mpg']
    videos = []

    # Get all files and child folders in the folder_path.
    files = os.listdir(folder_path)
    for name in files:
        relative_path = os.path.join(folder_path, name)
        if os.path.isfile(relative_path):
            absolute_path = os.path.abspath(relative_path)
            file_extension = os.path.splitext(absolute_path)[-1][1:]
            if file_extension in videos_type:
                videos.append(Video(name, absolute_path, file_extension))

    return videos


if __name__ == '__main__':
    folder_path = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/test/video/lab'
    images_parent_path = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology//data/test/video/lab/images'
    videos = get_all_videos(folder_path)

    for video in videos:
        video.cut_images(images_parent_path, 1)
        print(video)

    # x = change_size(
    #     'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology//data/test/video/lab/25.jpg')  # 得到文件名
    # cv2.imwrite(
    #     'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology//data/test/video/lab/25-crop.jpg', x)
