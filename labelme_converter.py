import os
import argparse

from converters.labelme2yolo import Converter2yolo
from converters.labelme2coco import Converter2coco
from converters.labelme2voc import Converter2voc


# Default args.
INPUT_PATH = 'labelme'
OUTPUT_PATH = 'current_dataset'
FORMAT = 'yolo'
CREATE_MARKED = False
POLYGONS = False

CONVERTERS = {'yolo': Converter2yolo,
              'coco': Converter2coco,
              'voc': Converter2voc}


def main():
    args = parse_args()
    # If ~ in the paths.
    args.input_path = os.path.expanduser(args.input_path)
    args.output_path = os.path.expanduser(args.output_path)

    converter_class = CONVERTERS[args.format]
    converter = converter_class(input_path=args.input_path,
                                output_path=args.output_path,
                                create_marked=args.create_marked,
                                polygons=args.polygons)
    converter.convert()


def parse_args():
    parser = argparse.ArgumentParser(description='Convert LabelMe dataset to Yolo / MsCOCO / PascalVOc')
    parser.add_argument('--input', dest='input_path', type=str,
                        help="Path with LabelMe dataset.", default=INPUT_PATH)
    parser.add_argument('--output', dest='output_path', type=str,
                        help="Path to save dataset.", default=OUTPUT_PATH)
    parser.add_argument('--format', dest='format', type=str,
                        help="Format to convert (voc, coco, yolo).", default=FORMAT)
    parser.add_argument('--create-marked', dest='create_marked', action='store_true',
                        help='Create marked images from dataset.', default=CREATE_MARKED)
    parser.add_argument('--poly', dest='polygons', action='store_true',
                        help='Create marked images with polygons.', default=POLYGONS)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
