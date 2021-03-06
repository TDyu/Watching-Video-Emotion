from PIL import Image
import os
import shutil

src_folder = "D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/test_black"
tar_folder = "D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/tar"
backup_folder = "D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/lab/video/backup"

rect_dict = {}

def isCrust(pix):
    return sum(pix) < 25


def hCheck(img, y, step=50):
    count = 0
    width = img.size[0]
    for x in range(0, width, step):
        if isCrust(img.getpixel((x, y))):
            count += 1
        if count > width / step / 2:
            return True
    return False


def vCheck(img, x, step=50):
    count = 0
    height = img.size[1]
    for y in range(0, height, step):
        if isCrust(img.getpixel((x, y))):
            count += 1
        if count > height / step / 2:
            return True
    return False


def boundaryFinder(img, crust_side, core_side, checker):
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


def handleImage_judge_zise(filename, tar):
    img = Image.open(os.path.join(src_folder, filename))
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size

    left = boundaryFinder(img, 0, width / 2, vCheck)
    right = boundaryFinder(img, width - 1, width / 2, vCheck)
    top = boundaryFinder(img, 0, height / 2, hCheck)
    bottom = boundaryFinder(img, height - 1, width / 2, hCheck)

    rect = (left, top, right, bottom)
    # print(rect)
    if rect not in rect_dict:
        rect_dict[rect] = 1
    else:
        rect_dict[rect] += 1


def handleImage_cut(filename, tar, size_tuple):
    img = Image.open(os.path.join(src_folder, filename))
    if img.mode != "RGB":
        img = img.convert("RGB")

    rect = size_tuple

    region = img.crop(rect)
    try:
        region.save(os.path.join(tar, filename), 'PNG')
    except SystemError:
        shutil.move(os.path.join(src_folder, filename),
                    os.path.join(tar, filename))


def folderCheck(foldername):
    if foldername:
        if not os.path.exists(foldername):
            os.mkdir(foldername)
            print("Info: Folder \"%s\" created" % foldername)
        elif not os.path.isdir(foldername):
            print("Error: Folder \"%s\" conflict" % foldername)
            return False
    return True


def juge_size():
    if folderCheck(tar_folder) and folderCheck(src_folder):
    # if folderCheck(tar_folder) and folderCheck(src_folder) and folderCheck(backup_folder):
        for filename in os.listdir(src_folder):
            if filename.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF"):
                handleImage_judge_zise(filename, tar_folder)
                # os.rename(os.path.join(src_folder, filename),
                #           os.path.join(backup_folder, filename))


def scorp_black(size_tuple):
    if folderCheck(tar_folder) and folderCheck(src_folder):
        for filename in os.listdir(src_folder):
            if filename.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF"):
                handleImage_cut(filename, tar_folder, size_tuple)


def find_max_times_rect():
    max_times = 0
    max_rect = None

    for key, value in rect_dict.items():
        if value > max_times:
            max_times = value
            max_rect = key
    
    # print(max_rect)
    # print(max_times)

    return max_rect


if __name__ == '__main__':
    juge_size()
    max_rect = find_max_times_rect()
    print(rect_dict)
    scorp_black(max_rect)

    # test = 'D:/Programming/Python/3.6/Project/WatchingVideoPsychophysiology/data/test/video/lab/24.jpg'
    # a = Image.open(test)
    # a.verify()
