import cv2
import numpy as np

from tqdm import tqdm
from typing import Tuple
from distutils.dir_util import copy_tree

from config import YOLO_NAME
from utils import ID2NAMES
from utils import SCRIPT_PATH
from utils import to_fixed
from utils import unit_lines_txt
from utils import coords_yolo2voc
from utils import safe_delete_file
from utils import make_empty_folder
from utils.images_marking import mark_image
from augmentations import augmentation_image
from utils.format_specifier import is_splitted


def augment_yolo(dataset_path: str, output_path: str, subsets: list, count: int) -> None:
    """
    Augment dataset or subset in Yolo format.

    :param dataset_path: path to dataset.
    :param output_path: path to save.
    :param subsets: subsets names or ['full'].
    :param count: count of augmented copy of image.
    """
    make_empty_folder(output_path)
    # Copy source dataset in `output_path` for augment it.
    copy_tree(dataset_path, output_path)
    # Update paths in dataset.data file.
    update_data_file(dataset_path, output_path)

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


def update_data_file(dataset_path: str, output_path: str) -> None:
    """
    Update paths in dataset.data file.

    :param dataset_path: old path to dataset.
    :param output_path: new path to dataset.
    """
    old_dir = dataset_path.split('/')[-1]
    new_dir = output_path.split('/')[-1]
    with open(f'{output_path}/{YOLO_NAME}/dataset.data', 'r') as rd:
        text = rd.read()
    text = text.replace(old_dir, new_dir)
    with open(f'{output_path}/{YOLO_NAME}/dataset.data', 'w') as wd:
        wd.write(text)


def augment_subsets(dataset_path: str, count: int, subsets: list) -> None:
    """
    Augment image in subsets of splitted dataset.

    :param dataset_path: path to dataset.
    :param count: count of augmented copy of image.
    :param subsets: subsets names or ['full'].
    """
    for subset in subsets:
        augment_subset(dataset_path, subset, count)

    # Update unused subsets.
    unused_sets = [s for s in ['train', 'test', 'val'] if s not in subsets]
    for unused_set in unused_sets:
        update_subset(dataset_path, unused_set)

    # Update trainval.txt.
    unit_lines_txt(first_path=f'{dataset_path}/{YOLO_NAME}/train.txt',
                   second_path=f'{dataset_path}/{YOLO_NAME}/val.txt',
                   out_path=f'{dataset_path}/{YOLO_NAME}/trainval.txt')


def update_subset(dataset_path: str, subset_name: str) -> None:
    """
    Update path in subset.

    :param dataset_path: path to dataset.
    :param subset_name: name of subset.
    """
    def replacer(s: str) -> str:
        old_parent_path = s.split(f'/{YOLO_NAME}/')[0]
        s.replace(old_parent_path, dataset_path)
        return s

    with open(f"{dataset_path}/{YOLO_NAME}/{subset_name}.txt", 'r') as rt:
        lines = rt.readlines()
    new_lines = list(map(replacer, lines))
    with open(f"{dataset_path}/{YOLO_NAME}/{subset_name}.txt", 'w') as wt:
        wt.writelines(new_lines)


def augment_subset(dataset_path: str, subset: str, count: int) -> None:
    """
    Augment single subset.

    :param dataset_path: path to dataset.
    :param subset: subset name (train, test, val).
    :param count: count of augmented copy of image.
    """
    txt_file = subset + '.txt'
    # Read all image filenames from subset txt file.
    img_filenames = get_imagenames_set(f'{dataset_path}/{YOLO_NAME}/{txt_file}')
    # Delete source txt file with image set.
    safe_delete_file(f'{dataset_path}/{YOLO_NAME}/{txt_file}')

    print(f"Augmentation {txt_file.split('.')[0]} set:")
    for img_name in tqdm(img_filenames):
        for i_image in range(count):
            augment_image(dataset_path, img_name, i_image, txt_file)
        # Delete source image and txt file with annotation.
        safe_delete_file(f"{dataset_path}/{YOLO_NAME}/dataset_data/{img_name.split('.')[0] + '.txt'}")
        safe_delete_file(f'{dataset_path}/{YOLO_NAME}/dataset_data/{img_name}')


def get_imagenames_set(file_set_path: str) -> list:
    """
    Read all image filenames from subset txt file.

    :param file_set_path: path to subset txt file.
    :return: list with names of images.
    """
    with open(file_set_path, 'r') as fr:
        images_paths = fr.readlines()
    filenames = []
    for image_path in images_paths:
        image_path = image_path.replace('\n', '')
        filenames.append(image_path.split('/')[-1])
    return filenames


def augment_image(dataset_path: str, img_filename: str, i_image: int, subset_filename: str) -> None:
    """
    Augment single image.

    :param dataset_path: path to dataset.
    :param img_filename: image filename.
    :param i_image: copy index.
    :param subset_filename: name of subset txt file.
    """
    image = cv2.imread(f'{dataset_path}/{YOLO_NAME}/dataset_data/{img_filename}')

    base_name, ext = img_filename.split('.')
    bboxes, categories = get_bbox_cat(dataset_path, base_name)

    # Augmentations.
    aug_dict = augmentation_image('yolo', image, bboxes, categories)
    aug_image = aug_dict['image']
    aug_bboxes = aug_dict['bboxes']
    aug_categories = aug_dict['classes_ids']

    # Save augmented image.
    save_path = f'{dataset_path}/{YOLO_NAME}/dataset_data/{base_name}_{i_image}.{ext}'
    cv2.imwrite(save_path, aug_image)

    # Save new annotation.
    new_annotation = create_ann(aug_bboxes, aug_categories)
    with open(f'{dataset_path}/{YOLO_NAME}/dataset_data/{base_name}_{i_image}.txt', 'w') as wt:
        wt.write(new_annotation)

    # Update new subset txt file.
    with open(f'{dataset_path}/{YOLO_NAME}/{subset_filename}', 'a') as at:
        at.write(f"{SCRIPT_PATH}/{dataset_path}/{YOLO_NAME}/dataset_data/{base_name}_{i_image}.{ext}\n")

    # Create new marked image.
    marked_image = create_mark_image(aug_dict)
    safe_delete_file(dataset_path + f'/MarkedImages/marked_{img_filename}')
    save_path = dataset_path + f'/MarkedImages/marked_{base_name}_{i_image}.{ext}'
    cv2.imwrite(save_path, marked_image)


def get_bbox_cat(dataset_path: str, base_name: str) -> Tuple[list, list]:
    """
    Get bboxes and categories from Yolo annotation.

    :param dataset_path: path to dataset.
    :param base_name: base name of image file.
    :return: lists of bboxes and categories.
    """
    txt_filename = base_name + '.txt'
    with open(f'{dataset_path}/{YOLO_NAME}/dataset_data/{txt_filename}', 'r') as tr:
        lines = tr.readlines()

    bboxes = []
    categories = []
    for line in lines:
        cat, x, y, w, h = [float(symb) for symb in line.replace('\n', '').split(' ')]
        categories.append(int(cat))
        bboxes.append([x, y, w, h])
    return bboxes, categories


def create_ann(bboxes: list, categories: list) -> str:
    """
    Create annotation in Yolo format.

    :param bboxes: bboxes of objects.
    :param categories: categories of objects.
    :return: string for write in txt file.
    """
    result = ''
    for cat, bbox in zip(categories, bboxes):
        bbox = list(map(to_fixed, bbox))
        result += f"{cat} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
    return result


def create_mark_image(aug_dict: dict) -> np.ndarray:
    """
    Create new marked image from augmented data.

    :param aug_dict: augmented data.
    :return: new marked image
    """
    # Create objects in main data (from labelme) format for marking.
    objects = []
    for bbox, cat_id in zip(aug_dict['bboxes'], aug_dict['classes_ids']):
        object = {'name': ID2NAMES[cat_id],
                  'points': coords_yolo2voc(bbox,
                                            img_w=aug_dict['image'].shape[1],
                                            img_h=aug_dict['image'].shape[0])}
        objects.append(object)
    marked_image = mark_image(aug_dict['image'], objects)
    return marked_image
