from config import VOC_NAME
from utils import get_xml_ann_data
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

    with open(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/{subset}.txt', 'r') as rt:
        base_names = [line for line in rt.read().split('\n') if line]

    subset_stat['images_count'] = len(base_names)

    for base_name in base_names:
        subset_stat = update_stat(dataset_path, subset_stat, base_name)

    return subset_stat


def update_stat(dataset_path: str, subset_stat: dict, base_name: str) -> dict:
    """
    Update data in statistic dictionary.

    :param dataset_path: path to dataset.
    :param subset_stat: statistic dictionary.
    :param base_name: base name of current image.
    :return: updated statistic dictionary.
    """
    xml_path = f'{dataset_path}/{VOC_NAME}/Annotations/{base_name}.xml'
    objects = get_xml_ann_data(xml_path)

    for obj in objects:
        label = obj['name']
        x_min, y_min, x_max, y_max = [int(coord) for coord in obj['bbox']]

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
