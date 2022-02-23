import os
import sys
import glob
import json
import shutil

import lxml.etree as ET
from pathlib import Path
from typing import Tuple

from config import COCO_NAME
from config import LABELS_FILE


SCRIPT_PATH = str(Path(__file__).parent.parent.absolute())
sys.path.append(SCRIPT_PATH)

# Get labels and labels ids.
LABELS_ID = {}
with open(f"{SCRIPT_PATH}/{LABELS_FILE}", 'r') as lr:
    LABELS = lr.read().split('\n')
for id, label in enumerate(LABELS):
    LABELS_ID[label] = id

ID2NAMES = {v: k for k, v in LABELS_ID.items()}


def to_fixed(num: float, digits=8) -> str:
    """
    Limits the number of decimal places.

    :param num: num.
    :param digits: a number of symbols after comma.
    :return:
    """
    return f"{num:.{digits}f}"


def make_empty_folder(out_path: str) -> None:
    """
    Make empty folder.

    :param out_path: path where the directory will be.
    """
    # Make path if not exists.
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Empty it if anything there.
    directories = os.listdir(out_path)
    for dir in directories:
        shutil.rmtree(f"{out_path}/{dir}", ignore_errors=True)
    files = glob.glob(os.path.join(out_path, '*.*'))
    for f in files:
        os.remove(f)


def copy_all_images(input_path: str, output_path: str, set: [] = None) -> None:
    """
    Copy all images from `input_path` to `output_path` with the ability to sort by a given set.

    :param input_path: path with images.
    :param output_path: path to copy.
    :param set: set with names of images to copy.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    exts = ['png', 'jpg', 'jpeg']
    files = os.listdir(input_path)
    for file in files:
        if file.split('.')[-1].lower() in exts and (set is None or file in set):
            file = f"/{file}"
            shutil.copy2(input_path + file, output_path + file)


def unit_lines_txt(first_path: str, second_path: str, out_path: str) -> None:
    """
    Write lines from two files in out file.

    :param first_path: path to first file.
    :param second_path: path to second file.
    :param out_path: path to out file.
    """
    lines = []
    for path in [first_path, second_path]:
        with open(path, 'r') as rt:
            lines.extend(rt.readlines())
    with open(out_path, 'w') as w:
        w.writelines(lines)


def safe_delete_file(path_to_file: str) -> None:
    """ Remove file if it exist. """
    if os.path.isfile(path_to_file):
        os.remove(path_to_file)


def coords_yolo2voc(coords: list, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Convert coord in Yolo format:
        `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
        `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
    To PascalVOC format:
        `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].

    :param x: normalized bbox center x coord.
    :param y: normalized bbox center y coord.
    :param w: normalized bbox width.
    :param h: normalized bbox height.
    :param img_w: image width.
    :param img_h: image height.
    :return: coords in PascalVOC format: x_min, y_min, x_max, y_max.
    """
    x, y, w, h = coords
    object_width = w * img_w
    object_height = h * img_h
    x_sum = x * img_w * 2
    y_sum = y * img_h * 2
    x_min = round((x_sum - object_width) / 2)
    y_min = round((y_sum - object_height) / 2)
    x_max = round(x_sum - x_min)
    y_max = round(y_sum - y_min)
    return x_min, y_min, x_max, y_max


def get_xml_ann_data(xml_path: str) -> list:
    """
    Parse objects data from XML annotation in PascalVOC format.

    :param xml_path: path to xml file.
    :return: list of objects data (dictionaries).
    """
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    objects = []
    for item in root:
        if item.tag == 'object':
            obj = {}
            for field in item:
                if field.tag == 'bndbox':
                    bbox = []
                    for coord in field:
                        bbox.append(coord.text)
                    obj['bbox'] = bbox
                else:
                    obj[field.tag] = field.text
            objects.append(obj)
    return objects


def read_json(dataset_path: str, filename: str) -> dict:
    """
    Read data from JSON file.

    :param dataset_path: path to dataset.
    :param filename: name of json file.
    :return: dictionary with data.
    """
    with open(f'{dataset_path}/{COCO_NAME}/Annotation/{filename}', 'rt', encoding='UTF-8') as annotations:
        coco_data = json.load(annotations)
    return coco_data
