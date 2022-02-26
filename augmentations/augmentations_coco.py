import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from typing import Tuple
from distutils.dir_util import copy_tree

from config import COCO_NAME
from utils import read_json
from utils import copy_all_images
from utils import safe_delete_file
from utils import make_empty_folder
from augmentations import augmentation_image
from utils.format_specifier import is_splitted
from utils.images_marking import mark_image
from utils.images_marking import create_mark_objects


def augment_coco(dataset_path: str, output_path: str, subsets: list, count: int) -> None:
    """
    Augment dataset or subset in MsCOCO format.

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
            augment_subset(output_path, subset='TrainTestVal', count=count)
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
        augment_subset(dataset_path, subset.capitalize(), count)


def augment_subset(dataset_path: str, subset: str, count: int) -> None:
    """
    Augment all images in subset.

    :param dataset_path: path to dataset.
    :param subset: name of subset (train, test, val).
    :param count: count copies of image.
    """
    img_filenames = os.listdir(f'{dataset_path}/{COCO_NAME}/{subset}')
    # Read data from json file.
    coco_data = read_json(dataset_path, subset + '.json')

    print(f"Augmentation {subset} set:")
    for img_filename in tqdm(img_filenames):
        for i_image in range(count):
            coco_data = augment_image(dataset_path, coco_data, img_filename, i_image + 1, subset)
        # Delete image from subset directory.
        safe_delete_file(f'{dataset_path}/{COCO_NAME}/{subset}/{img_filename}')
        # Delete image from TrainVal subset directory.
        if subset in ['Train', 'Val']:
            safe_delete_file(f'{dataset_path}/{COCO_NAME}/TrainVal/{img_filename}')

    # Delete data of old images.
    coco_data = filter_coco(img_filenames, coco_data)
    # Save new coco data.
    with open(f'{dataset_path}/{COCO_NAME}/Annotation/{subset}.json', 'w') as outfile:
        json.dump(coco_data, outfile, indent=2)

    # Update TRainVal json file.
    if subset in ['Train', 'Val']:
        update_trainval(dataset_path, coco_data, img_filenames, subset)


def augment_image(dataset_path: str, coco_data: dict, img_filename: str, i_image: int, subset: str) -> dict:
    """
    Augment single image.

    :param dataset_path: path to dataset.
    :param coco_data: data in coco format.
    :param img_filename: image file name.
    :param i_image: index of image copy.
    :param subset: directory name of subset.
    """
    image = cv2.imread(f"{dataset_path}/{COCO_NAME}/{subset}/{img_filename}")

    images, annotations, categories = coco_data['images'], coco_data['annotations'], coco_data['categories']
    i_image_json = get_img_json_id(img_filename, images)
    objects_data = [ann for ann in annotations if ann['image_id'] == i_image_json]
    bboxes, categories_ids, objects_ids = prepare_objects(objects_data)

    # Augmentations.
    aug_dict = augmentation_image('coco', image, bboxes, categories_ids)
    aug_image = aug_dict['image']
    aug_bboxes = [[round(coord) for coord in bbox] for bbox in aug_dict['bboxes']]
    aug_categories_ids = aug_dict['classes_ids']

    # Save augmented image.
    new_imagename = img_filename.split('.')[0] + f'_{i_image}.' + img_filename.split('.')[1]
    cv2.imwrite(f"{dataset_path}/{COCO_NAME}/{subset}/{new_imagename}", aug_image)

    # Create new marked image.
    create_mark_image(dataset_path, aug_image, aug_bboxes, aug_categories_ids, img_filename, new_imagename)

    # Create new coco data.
    coco_data = create_new_coco(images=images,
                                annotations=annotations,
                                categories=categories,
                                aug_categories_ids=aug_categories_ids,
                                aug_bboxes=aug_bboxes,
                                objects_ids=objects_ids,
                                i_image=i_image,
                                i_image_json=i_image_json,
                                new_imagename=new_imagename,
                                image=image)
    return coco_data


def get_img_json_id(img_filename: str, images_data: list) -> int:
    """
    Get id of image from json annotation.

    :param img_filename:
    :param images_data:
    :return:
    """
    id_list = [image_data['id'] for image_data in images_data if image_data['file_name'] == img_filename]
    if not id_list:
        raise Exception(f"Image annotation {img_filename} not founded!")
    return id_list[0]


def prepare_objects(objects_data: list) -> Tuple[list, list, list]:
    """
    Split `objects_data` on `categories_ids`, `objects_ids`, `bboxes`.

    :param objects_data: data of objects on image.
    :return: splitted data of objects.
    """
    bboxes = []
    categories_ids = []
    objects_ids = []
    for obj in objects_data:
        categories_ids.append(obj['category_id'])
        objects_ids.append(obj['id'])
        bboxes.append([int(coord) for coord in obj['bbox']])
    return bboxes, categories_ids, objects_ids


def create_mark_image(dataset_path: str, aug_image: np.ndarray, aug_bboxes: list, aug_categories_ids: list,
                      old_filename: str, new_filename: str) -> None:
    """
    Create new marked image.

    :param dataset_path: path to dataset.
    :param aug_image: augmented image.
    :param aug_bboxes: augmented bboxes.
    :param aug_categories_ids: augmented categories ids.
    :param old_filename: old image filename.
    :param new_filename: new image filename.
    """
    decrement_aug_cat_ids = [cat_id - 1 for cat_id in aug_categories_ids]  # category ids -1, start with 0.
    # From [x_min, y_min, obj_w, obj_h] to [x_min, y_min, x_max, y_max].
    voc_bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in aug_bboxes]
    # Create objects in format for marking.
    objects = create_mark_objects(voc_bboxes, decrement_aug_cat_ids)
    marked_image = mark_image(aug_image, objects)
    # Delete old image and save new image.
    safe_delete_file(f'{dataset_path}/MarkedImages/marked_{old_filename}')
    save_path = f'{dataset_path}/MarkedImages/marked_{new_filename}'
    cv2.imwrite(save_path, marked_image)


def create_new_coco(images: list,
                    annotations: list,
                    categories: list,
                    aug_categories_ids: list,
                    aug_bboxes: list,
                    objects_ids: list,
                    i_image: int,
                    i_image_json: int,
                    new_imagename: str,
                    image: np.ndarray) -> dict:
    """
    Create new coco annotation.

    :param images: images annotations.
    :param annotations: annotations of objects.
    :param categories: categories annotations.
    :param aug_categories_ids: augmented categories ids.
    :param aug_bboxes: augmented bboxes.
    :param objects_ids: ids of objects from json file.
    :param i_image: index augmented copy of image.
    :param i_image_json: id of image from json annotation.
    :param new_imagename: new image name with index.
    :param image: source image.
    :return: dictionary with new coco annotation.
    """
    for cat_id, bbox, src_id in zip(aug_categories_ids, aug_bboxes, objects_ids):
        annotations.append({"id": src_id + i_image,
                            "image_id": i_image_json + i_image,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "category_id": cat_id,
                            "segmentation": []})

    images.append({"file_name": new_imagename,
                   "height": image.shape[0],
                   "width": image.shape[1],
                   "id": i_image_json + i_image})

    coco_data = {'images': images,
                 'categories': categories,
                 'annotations': annotations}
    return coco_data


def filter_coco(images_names: list, coco_data: dict) -> dict:
    """
    Delete data about images from `images_names` in `coco_data` annotations.

    :param images_names: name of old images.
    :param coco_data: coco annotations.
    :return: coco annotations without old data.
    """
    images_ids = [im_data['id'] for im_data in coco_data['images'] if im_data['file_name'] in images_names]
    new_images = [im_data for im_data in coco_data['images'] if im_data['file_name'] not in images_names]
    new_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] not in images_ids]
    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations
    return coco_data


def update_trainval(dataset_path: str, coco_data: dict, img_filenames: list, subset: str) -> None:
    """
    Update TrainVal subset.

    :param dataset_path: path to dataset.
    :param coco_data: coco annotations of current subset (Train or Val).
    :param img_filenames: name of images from current subset.
    :param subset: current subset.
    """
    coco_trainval = read_json(dataset_path, 'TrainVal.json')
    coco_trainval = filter_coco(img_filenames, coco_trainval)
    coco_trainval = update_coco_trainval(coco_trainval, coco_data)
    with open(f'{dataset_path}/{COCO_NAME}/Annotation/TrainVal.json', 'w') as outfile:
        json.dump(coco_trainval, outfile, indent=2)
    copy_all_images(f'{dataset_path}/{COCO_NAME}/{subset}', f'{dataset_path}/{COCO_NAME}/TrainVal')


def update_coco_trainval(coco_trainval: dict, coco_data: dict) -> dict:
    """
    Update coco annotation from TrainVal.json.

    :param coco_trainval: coco annotation from TrainVal.json.
    :param coco_data: coco annotation from current subset.
    :return: updated coco annotation for TrainVal.json.
    """
    coco_trainval['images'].extend(coco_data['images'])
    coco_trainval['annotations'].extend(coco_data['annotations'])
    return coco_trainval
