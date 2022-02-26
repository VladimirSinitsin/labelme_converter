import os
import json

from tqdm import tqdm

from config import COCO_NAME
from utils import read_json
from resizing import get_ratio
from resizing import resize_image
from utils.format_specifier import is_splitted


def resize(dataset_path: str, new_h: int, new_w: int) -> None:
    """
    Resize all image in MsCOCO format dataset.

    :param dataset_path: path to dataset.
    :param new_h: new height of images.
    :param new_w: new width of images.
    """
    if is_splitted(dataset_path):
        set_names = ['Train', 'Test', 'Val', 'TrainVal']
    else:
        set_names = ['TrainTestVal']
    # Resize sets.
    for set_name in set_names:
        resize_set(dataset_path, set_name, new_h, new_w)


def resize_set(dataset_path: str, set_name: str, new_h: int, new_w: int) -> None:
    """
    Resize all images in set of dataset.

    :param dataset_path: path to dataset.
    :param set_name: name of set.
    :param new_h: new height of images.
    :param new_w: new width of images.
    """
    images_filenames = os.listdir(f'{dataset_path}/{COCO_NAME}/{set_name}')
    coco_data = read_json(dataset_path, set_name + '.json')

    print(f"Resize images in {set_name} set:")
    for img_filename in tqdm(images_filenames):
        # Find image id and write new height and width in annotation.
        img_ids = [(index, img_data['id'])
                   for index, img_data in enumerate(coco_data['images'])
                   if img_data['file_name'] == img_filename]
        if not img_ids:
            raise Exception(f"Image {img_filename} not found in MsCOCO data!")
        img_data_index, img_id = img_ids[0]
        coco_data['images'][img_data_index]['height'] = new_h
        coco_data['images'][img_data_index]['width'] = new_w
        # Resize objects annotations.
        ratio_w, ratio_h = get_ratio(f'{dataset_path}/{COCO_NAME}/{set_name}/{img_filename}', new_w, new_h)
        objects_inds = [index for index, obj in enumerate(coco_data['annotations']) if obj['image_id'] == img_id]
        for ind in objects_inds:
            coco_data['annotations'][ind] = resize_object(coco_data['annotations'][ind], ratio_w, ratio_h)
        # Write resized images.
        resize_image(f'{dataset_path}/{COCO_NAME}/{set_name}/{img_filename}', new_w, new_h)
        resize_image(f'{dataset_path}/MarkedImages/marked_{img_filename}', new_w, new_h)
    # Write resized images annotation.
    with open(f'{dataset_path}/{COCO_NAME}/Annotation/{set_name}.json', 'w') as outfile:
        json.dump(coco_data, outfile, indent=2)


def resize_object(object: dict, ratio_w: float, ratio_h: float) -> dict:
    """
    Resize object in annotation.

    :param object: data of object.
    :param ratio_w: ratio width.
    :param ratio_h: ratio height.
    :return: new object data.
    """
    x, y, w, h = object['bbox']
    object['bbox'] = round(x * ratio_w), round(y * ratio_h), round(w * ratio_w), round(h * ratio_h)
    object['area'] = round(w * ratio_w) * round(h * ratio_h)
    return object
