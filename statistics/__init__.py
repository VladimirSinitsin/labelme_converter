import os

from prettytable import PrettyTable  # pip install PTable

from utils import LABELS
from utils import ID2NAMES
from utils import to_fixed


def print_stat(statistics: dict, save_txt: bool, path_to_save: str = None) -> None:
    """
    Print statistic on console and save on txt file.

    :param statistics: statistics of subsets.
    :param save_txt: if you need to save statistic in txt file.
    :param path_to_save: path to save txt file.
    """
    # Delete old stat.txt file.
    if os.path.isfile(path_to_save + '/stat.txt'):
        os.remove(path_to_save + '/stat.txt')

    for subset in statistics.keys():
        stat = statistics[subset]

        # Create table with statistic.
        table = PrettyTable()
        table.title = subset + f" | count of images: {stat['images_count']}"
        table.field_names = ['class', 'number of objects', 'avg_width', 'avg_height', 'avg_area']
        for label in LABELS:
            subrow = get_row(stat[label]) if label in stat.keys() else [0, 0, 0, 0]
            label = ID2NAMES[label] if not isinstance(label, str) else label
            table.add_row([label] + subrow)
        print(table)
        print()

        if save_txt:
            write_txt(path_to_save, table)


def get_row(stat_label: dict) -> list:
    """
    Get row of table.

    :param stat_label: statistic of label in subset.
    :return: row of table.
    """
    count = stat_label['count']
    avg_width = to_fixed(stat_label['sum_width'] / stat_label['count'], digits=2)
    avg_height = to_fixed(stat_label['sum_height'] / stat_label['count'], digits=2)
    avg_area = to_fixed(stat_label['sum_area'] / stat_label['count'], digits=2)
    return [count, avg_width, avg_height, avg_area]


def write_txt(path_to_save: str, table: PrettyTable) -> None:
    """
    Write statistic in txt file.

    :param path_to_save: path to save txt file.
    :param table: table with statistic.
    """
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    with open(path_to_save + '/stat.txt', 'a') as file:
        table_txt = table.get_string()
        file.write(table_txt)
        file.write(os.linesep)
