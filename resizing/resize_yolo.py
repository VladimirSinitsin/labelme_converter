import os
import cv2

from tqdm import tqdm

from config import YOLO_NAME
from resizing import resize_image
from utils import coords_yolo2voc
from converters.labelme2yolo import Converter2yolo


def resize(dataset_path: str, new_h: int, new_w: int) -> None:
    """
    Resize all images and annotations in Yolo dataset.

    :param dataset_path: path to dataset.
    :param new_h: new height of images.
    :param new_w: new width of images.
    """
    # If .png or .jpg or .JPEG.
    img_names = [file for file in os.listdir(f'{dataset_path}/{YOLO_NAME}/dataset_data') if not file.endswith('.txt')]
    print('Resize:')
    for img_filename in tqdm(img_names):
        resize_image_data(dataset_path, img_filename, new_w, new_h)


def resize_image_data(dataset_path: str, img_filename: str, new_w: int, new_h: int) -> None:
    """
    Resize image and annotation.

    :param dataset_path: path to dataset.
    :param img_filename: image filename.
    :param new_w: new width of image.
    :param new_h: new height of image.
    """
    ann_file = img_filename.replace(img_filename.split('.')[-1], 'txt')
    img_path = f'{dataset_path}/{YOLO_NAME}/dataset_data/{img_filename}'
    ann_path = f'{dataset_path}/{YOLO_NAME}/dataset_data/{ann_file}'
    viz_path = f'{dataset_path}/MarkedImages/marked_{img_filename}'

    resize_annotation(ann_path, img_path, new_w, new_h)

    resize_image(img_path, new_w, new_h)
    resize_image(viz_path, new_w, new_h)


def resize_annotation(ann_path: str, img_path: str, new_w: int, new_h: int) -> None:
    """
    Resize annotation of image.

    :param ann_path: path to txt file.
    :param img_path: path to image
    :param new_h: new height of image.
    :param new_w: new width of image.
    :return:
    """
    image = cv2.imread(img_path)
    img_h, img_w, _ = image.shape
    ratio_w = new_w / img_w
    ratio_h = new_h / img_h
    # Read objects annotations.
    with open(ann_path, 'r') as rt:
        objects = [line for line in rt.read().split('\n') if line]  # without empty lines
    # Resize annotations.
    new_objects = []
    for obj in objects:
        new_obj = resize_object(obj, ratio_w, ratio_h, img_w, img_h, new_w, new_h)
        new_objects.append(new_obj)
    # Write in txt file.
    with open(ann_path, 'w') as wt:
        wt.writelines(new_objects)


def resize_object(obj: str, ratio_w: float, ratio_h: float, img_w: int, img_h: int, new_w: int, new_h: int) -> str:
    """
    Resize coordinates of object.

    :param obj: line with data about object.
    :param ratio_w: relationship between new and old width.
    :param ratio_h: relationship between new and old height.
    :param img_w: image width.
    :param img_h: image height.
    :param new_w: new width.
    :param new_h: new height.
    :return: line for write with resized object.
    """
    cat, x, y, w, h = [float(value) for value in obj.split(' ') if value]
    x_min, y_min, x_max, y_max = coords_yolo2voc([x, y, w, h], img_w, img_h)
    # Resize.
    x_min, y_min, x_max, y_max = x_min * ratio_w, y_min * ratio_h, x_max * ratio_w, y_max * ratio_h
    x, y, w, h = Converter2yolo.convert_coords([x_min, y_min, x_max, y_max], new_w, new_h)
    return f'{int(cat)} {x} {y} {w} {h}\n'
