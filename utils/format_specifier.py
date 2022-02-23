import os

from config import VOC_NAME
from config import COCO_NAME
from config import YOLO_NAME


def format_name(dataset_path: str) -> str:
    """
    Define format name of dataset (voc, coco, yolo).

    :param dataset_path: path to dataset.
    :return: dataset short name.
    """
    if os.path.isdir(f'{dataset_path}/{VOC_NAME}'):
        return 'voc'
    if os.path.isdir(f'{dataset_path}/{COCO_NAME}'):
        return 'coco'
    if os.path.isdir(f'{dataset_path}/{YOLO_NAME}'):
        return 'yolo'
    else:
        raise Exception(f'Dataset format in "{dataset_path}" not found!')


def is_splitted(dataset_path: str) -> bool:
    """
    Determines if the dataset has been split.

    :param dataset_path: path to dataset.
    :return: dataset has been split or not.
    """
    # If a file with unsplit data is contained, then it has not been split.
    if format_name(dataset_path) == 'voc':
        return not os.path.isfile(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/train_test_val.txt')
    if format_name(dataset_path) == 'coco':
        return not os.path.isfile(f'{dataset_path}/{COCO_NAME}/Annotation/TrainTestVal.json')
    if format_name(dataset_path) == 'yolo':
        return not os.path.isfile(f'{dataset_path}/{YOLO_NAME}/train_test_val.txt')
