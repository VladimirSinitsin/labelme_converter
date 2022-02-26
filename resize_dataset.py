import argparse

from distutils.dir_util import copy_tree

from utils import make_empty_folder
from utils.format_specifier import format_name
from resizing.resize_voc import resize as resize_voc
from resizing.resize_yolo import resize as resize_yolo
from resizing.resize_coco import resize as resize_coco


# Default args.
INPUT_PATH = 'augmented_dataset'
OUTPUT_PATH = 'resized_dataset'
NEW_W = 512
NEW_H = 640


def main():
    args = parse_args()

    make_empty_folder(args.output_path)
    copy_tree(args.input_path, args.output_path)

    format = format_name(args.output_path)
    if format == 'yolo':
        resize_yolo(args.output_path, args.new_w, args.new_h)
    elif format == 'coco':
        resize_coco(args.output_path, args.new_w, args.new_h)
    elif format == 'voc':
        resize_voc(args.output_path, args.new_w, args.new_h)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Resize Yolo/MsCOCO/PascalVOc dataset on one shape.')
    parser.add_argument('--input', dest='input_path', type=str,
                        help="Path with dataset.", default=INPUT_PATH)
    parser.add_argument('--output', dest='output_path', type=str,
                        help="Path to save resized dataset.", default=OUTPUT_PATH)
    parser.add_argument('--new_w', dest='new_w', type=int,
                        help="New width of images.", default=NEW_W)
    parser.add_argument('--new_h', dest='new_h', type=int,
                        help="New height of images.", default=NEW_H)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
