import cv2

from config import YOLO_NAME
from utils import ID2NAMES
from utils import coords_yolo2voc
from utils.format_specifier import is_splitted


def get_stat_dict(dataset_path: str) -> dict:
    """
    Get dictionary with statistic of all subsets.

    :param dataset_path: path to dataset.
    :return: dictionary with statistic of all subsets.
    """
    if is_splitted(dataset_path):
        subsets = ['train', 'test', 'val', 'trainval']
    else:
        subsets = ['train_test_val']
    stat_dict = {}
    for subset in subsets:
        stat_dict[subset] = get_subset_stat(dataset_path, subset)
    return stat_dict


def get_subset_stat(dataset_path: str, subset: str) -> dict:
    """
    Get dictionary with statistic of current subset.

    :param dataset_path: path to dataset.
    :param subset: name of subset.
    :return: dictionary with statistic of current subset.
    """
    subset_stat = {}

    with open(f'{dataset_path}/{YOLO_NAME}/{subset}.txt', 'r') as rt:
        images_paths = [line for line in rt.read().split('\n') if line]

    subset_stat['images_count'] = len(images_paths)

    images_names = [path.split('/')[-1] for path in images_paths if path]
    for img_name in images_names:
        subset_stat = update_stat(dataset_path, subset_stat, img_name)
    return subset_stat


def update_stat(dataset_path: str, subset_stat: dict, img_name: str) -> dict:
    """
    Update data in statistic dictionary.

    :param dataset_path: path to dataset.
    :param subset_stat: statistic dictionary.
    :param img_name: name of current image.
    :return: updated statistic dictionary.
    """
    image = cv2.imread(f'{dataset_path}/{YOLO_NAME}/dataset_data/{img_name}')
    img_w = image.shape[1]
    img_h = image.shape[0]

    base_name = img_name.split('.')[0]
    with open(f'{dataset_path}/{YOLO_NAME}/dataset_data/{base_name}.txt', 'r') as rt:
        annotations = rt.read().split('\n')
    for ann in [item for item in annotations if item]:
        label_id, x, y, w, h = [float(item) for item in ann.split(' ') if item]
        x_min, y_min, x_max, y_max = coords_yolo2voc([x, y, w, h], img_w, img_h)

        label = ID2NAMES[label_id]
        if label not in subset_stat.keys():
            subset_stat[label] = {'count': 1,
                                  'sum_width': x_max - x_min,
                                  'sum_height': y_max - y_min,
                                  'sum_area': (x_max - x_min) * (y_max - y_min)}
        else:
            subset_stat[label]['count'] += 1
            subset_stat[label]['sum_width'] += x_max - x_min
            subset_stat[label]['sum_height'] += y_max - y_min
            subset_stat[label]['sum_area'] += (x_max - x_min) * (y_max - y_min)

    return subset_stat
