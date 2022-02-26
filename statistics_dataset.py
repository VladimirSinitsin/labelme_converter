import argparse

from utils.format_specifier import format_name
from statistics import print_stat
from statistics.stat_voc import get_stat_dict as stat_voc
from statistics.stat_coco import get_stat_dict as stat_coco
from statistics.stat_yolo import get_stat_dict as stat_yolo


# Default args.
INPUT_PATH = 'splitted_dataset',
SAVE = False,
SAVE_PATH = '###'  # '###' = in dataset dir


def main():
    args = parse_args()

    save_path = args.save_path if args.save_path != '###' else args.input_path

    stat(format=format_name(args.input_path),
         input_path=args.input_path,
         save=args.save,
         save_path=save_path)


def stat(format: str, input_path: str, save: bool, save_path: str) -> None:
    statistics = {}
    if format == 'yolo':
        statistics = stat_yolo(input_path)
    if format == 'voc':
        statistics = stat_voc(input_path)
    if format == 'coco':
        statistics = stat_coco(input_path)

    print_stat(statistics, save, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Statistic of Yolo/MsCOCO/PascalVOc dataset.')
    parser.add_argument('--input', dest='input_path', type=str,
                        help="Path with dataset.", default=INPUT_PATH)
    parser.add_argument('--save', dest='save', action='store_true',
                        help='Save statistic to txt file.', default=SAVE)
    parser.add_argument('--save_path', dest='save_path', type=str,
                        help="Path to save statistic.", default=SAVE_PATH)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
