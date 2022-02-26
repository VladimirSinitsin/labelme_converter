import os

from tqdm import tqdm

from config import VOC_NAME
from resizing import get_ratio
from resizing import resize_image
from utils import get_xml_ann_data
from augmentations.augmentations_voc import save_new_xml


def resize(dataset_path: str, new_w: int, new_h: int) -> None:
    """
    Resize all image in PascalVOC format dataset.

    :param dataset_path: path to dataset.
    :param new_w: new width of images.
    :param new_h: new height of images.
    """
    images_filenames = os.listdir(f'{dataset_path}/{VOC_NAME}/JPEGImages')
    print('Resize:')
    for image_filename in tqdm(images_filenames):
        resize_image_data(dataset_path, image_filename, new_w, new_h)


def resize_image_data(dataset_path: str, image_filename: str, new_w: int, new_h: int) -> None:
    """
    Resize image anf annotation.

    :param dataset_path: path to dataset.
    :param image_filename: image filename.
    :param new_w: new width of image.
    :param new_h: new height of image.
    """
    img_path = f'{dataset_path}/{VOC_NAME}/JPEGImages/{image_filename}'
    viz_path = f'{dataset_path}/MarkedImages/marked_{image_filename}'
    xml_path = f"{dataset_path}/{VOC_NAME}/Annotations/{image_filename.split('.')[0] + '.xml'}"

    objects = get_xml_ann_data(xml_path)

    # Resize annotation.
    new_objects = resize_objects(objects, img_path, new_w, new_h)
    save_new_xml(dataset_path, image_filename, img_hwd=(new_h, new_w, 3), objects=new_objects)

    # Resize image and marked image.
    resize_image(img_path, new_w, new_h)
    resize_image(viz_path, new_w, new_h)


def resize_objects(objects: list, img_path: str, new_w: int, new_h: int) -> list:
    """
    Resize objects coordinates in annotation.

    :param objects: objects annotations.
    :param img_path: path to image.
    :param new_w: new width of image.
    :param new_h: new height of image.
    :return: resized objects in format for save in xml.
    """
    ratio_w, ratio_h = get_ratio(img_path, new_w, new_h)
    new_objects = []
    for obj_ind, obj_ann in enumerate(objects):
        x_min, y_min, x_max, y_max = [int(coord) for coord in obj_ann['bbox']]
        obj = {
            'points': [round(x_min * ratio_w), round(y_min * ratio_h), round(x_max * ratio_w), round(y_max * ratio_h)],
            'name': obj_ann['name']}
        new_objects.append(obj)
    return new_objects
