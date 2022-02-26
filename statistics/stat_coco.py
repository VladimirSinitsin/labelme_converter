import json

from utils import ID2NAMES
from config import COCO_NAME
from utils.format_specifier import is_splitted


def get_stat_dict(dataset_path: str) -> dict:
    """
    Get dictionary with statistic of all subsets.

    :param dataset_path: path to dataset.
    :return: dictionary with statistic of all subsets.
    """
    if is_splitted(dataset_path):
        subsets = ['Train', 'Test', 'Val', 'TrainVal']
    else:
        subsets = ['TrainTestVal']
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
    subset_stat = {'images_ids': set()}

    with open(f'{dataset_path}/{COCO_NAME}/Annotation/{subset}.json', 'rt', encoding='UTF-8') as annotations:
        coco_data = json.load(annotations)
    for ann in coco_data['annotations']:
        # `subset_stat['images_ids']` is needed to count `subset_stat['images_count']`.
        subset_stat['images_ids'].add(ann['image_id'])
        label_id = ann['category_id'] - 1  # categories in coco start with 1.
        label = ID2NAMES[label_id]
        if label not in subset_stat.keys():
            subset_stat[label] = {'count': 1,
                                  'sum_width': ann['bbox'][2],
                                  'sum_height': ann['bbox'][3],
                                  'sum_area': ann['area']}
        else:
            subset_stat[label]['count'] += 1
            subset_stat[label]['sum_width'] += ann['bbox'][2]
            subset_stat[label]['sum_height'] += ann['bbox'][3]
            subset_stat[label]['sum_area'] += ann['area']
    subset_stat['images_count'] = len(subset_stat['images_ids'])
    return subset_stat
