import os
import cv2
import fnmatch
import numpy as np
import lxml.etree as ET

from tqdm import tqdm
from typing import Tuple
from distutils.dir_util import copy_tree

from config import VOC_NAME
from utils import LABELS_ID
from utils import unit_lines_txt
from utils import safe_delete_file
from utils import get_xml_ann_data
from utils import make_empty_folder
from utils.images_marking import mark_image
from utils.images_marking import create_mark_objects
from augmentations import augmentation_image
from utils.format_specifier import is_splitted
from converters.labelme2voc import Converter2voc


def augment_voc(dataset_path: str, output_path: str, subsets: list, count: int) -> None:
    """
    Augment dataset or subset in PascalVOC format.

    :param dataset_path: path to dataset.
    :param output_path: path to save.
    :param subsets: subsets names or ['full'].
    :param count: count of augmented copy of image.
    """
    make_empty_folder(output_path)
    # Copy source dataset in `output_path` for augment it.
    copy_tree(dataset_path, output_path)

    if is_splitted(output_path):
        if 'full' in subsets:
            augment_subsets(output_path, count, subsets=['train', 'test', 'val'])
        elif set(subsets).issubset({'train', 'test', 'val'}):
            augment_subsets(output_path, count, subsets=subsets)
        else:
            raise Exception(f"Subsets must be inside a ['full', 'train', 'test', 'val']!")
    else:
        if 'full' in subsets or (not subsets):
            augment_subset(output_path, subset='train_test_val', count=count)
        elif set(subsets).issubset({'train', 'test', 'val'}):
            raise Exception(f"For augment subset dataset must be split! Set `--full` flag.")
        else:
            raise Exception(f"Subsets must be inside a ['full', 'train', 'test', 'val']!")


def augment_subsets(dataset_path: str, count: int, subsets: list) -> None:
    """
    Augment image in subsets of splitted dataset.

    :param dataset_path: path to dataset.
    :param count: count of augmented copy of image.
    :param subsets: list of subsets for augmentation.
    """
    for subset in subsets:
        augment_subset(dataset_path, subset, count)

    # Update trainval.txt.
    unit_lines_txt(first_path=f'{dataset_path}/{VOC_NAME}/ImageSets/Main/train.txt',
                   second_path=f'{dataset_path}/{VOC_NAME}/ImageSets/Main/val.txt',
                   out_path=f'{dataset_path}/{VOC_NAME}/ImageSets/Main/trainval.txt')


def augment_subset(dataset_path: str, subset: str, count: int) -> None:
    """
    Augment all images in subset.

    :param dataset_path: path to dataset.
    :param subset: name of subset (train, test, val).
    :param count: count copies of image.
    """
    txt_filename = subset + '.txt'
    # Read all image filenames from subset txt file.
    img_filenames = get_imagenames_set(dataset_path, txt_filename)
    # Delete source txt file with image set.
    safe_delete_file(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/{txt_filename}')

    print(f"Augmentation {subset} set:")
    for img_filename in tqdm(img_filenames):
        for i_image in range(count):
            augment_image(dataset_path, img_filename, i_image, txt_filename)
        # Delete source image and txt file with annotation.
        safe_delete_file(f"{dataset_path}/{VOC_NAME}/Annotations/{img_filename.split('.')[0] + '.xml'}")
        safe_delete_file(f'{dataset_path}/{VOC_NAME}/JPEGImages/{img_filename}')


def get_imagenames_set(dataset_path: str, txt_filename: str) -> list:
    """
    Get images names from subset txt file with base names.

    :param dataset_path: path to dataset.
    :param txt_filename: subset txt file.
    :return: list of images names.
    """
    with open(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/{txt_filename}', 'r') as rt:
        basenames = rt.read().split('\n')[:-1]
    images_names = []
    for basename in basenames:
        # Find first image start with basename.
        image_name = fnmatch.filter(os.listdir(f'{dataset_path}/{VOC_NAME}/JPEGImages'), f'{basename}*')[0]
        images_names.append(image_name)
    return images_names


def augment_image(dataset_path: str, img_filename: str, i_image: int, subset_filename: str) -> None:
    """
    Augment single image.

    :param dataset_path: path to dataset.
    :param img_filename: image file name.
    :param i_image: index of image copy.
    :param subset_filename: filename of subset with .txt ext.
    """
    # Read image.
    image = cv2.imread(f'{dataset_path}/{VOC_NAME}/JPEGImages/{img_filename}')

    # Get bboxes and labels of objects on image.
    base_name, ext = img_filename.split('.')
    new_img_filename = f'{base_name}_{i_image}.{ext}'
    bboxes, labels = get_bbox_labels(dataset_path, base_name)

    # Augmentations.
    aug_dict = augmentation_image('pascal_voc', image, bboxes, labels)
    aug_image = aug_dict['image']
    aug_bboxes = [[round(coord) for coord in bbox] for bbox in aug_dict['bboxes']]
    aug_labels_idx = aug_dict['classes_ids']

    # Save augmented image.
    save_path = f'{dataset_path}/{VOC_NAME}/JPEGImages/{new_img_filename}'
    cv2.imwrite(save_path, aug_image)

    # Create objects in special format (list of dictionaries).
    objects = create_mark_objects(aug_bboxes, aug_labels_idx)
    # Save new annotation.
    save_new_xml(dataset_path, new_img_filename, image.shape, objects)

    # Update new subset txt file.
    with open(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/{subset_filename}', 'a') as at:
        at.write(f'{base_name}_{i_image}\n')

    # Create new marked image.
    marked_image = mark_image(aug_image, objects)
    safe_delete_file(dataset_path + f'/MarkedImages/marked_{img_filename}')
    save_path = dataset_path + f'/MarkedImages/marked_{base_name}_{i_image}.{ext}'
    cv2.imwrite(save_path, marked_image)


def get_bbox_labels(dataset_path: str, base_name: str) -> Tuple[list, list]:
    """
    Get bboxes and labels from xml file.

    :param dataset_path: path to dataset.
    :param base_name: base name of image.
    :return: bboxes and labels.
    """
    objects = get_xml_ann_data(f'{dataset_path}/{VOC_NAME}/Annotations/{base_name}.xml')

    bboxes = []
    labels = []
    for obj in objects:
        label = obj['name']
        labels.append(LABELS_ID[label])
        bboxes.append([int(coord) for coord in obj['bbox']])
    return bboxes, labels


def save_new_xml(dataset_path: str, img_name: str, img_hwd: Tuple[int, int, int], objects: list) -> None:
    """
    Create and save xml file with annotation.

    :param dataset_path: path to dataset.
    :param img_name: filename of image.
    :param img_hwd: height, width, depth of image.
    :param objects: list with data of objects.
    """
    xml_tree = Converter2voc.create_file(img_name, img_hwd, objects)
    # Write annotation file.
    xml_path = f"{dataset_path}/{VOC_NAME}/Annotations/{img_name.split('.')[0]}.xml"
    with open(xml_path, 'wb') as f:
        # f.write('<?xml version="1.0" encoding="utf-8"?>\n'.encode())
        f.write(ET.tostring(xml_tree, pretty_print=True))
