import argparse

from utils.format_specifier import format_name
from augmentations.augmentations_voc import augment_voc
from augmentations.augmentations_yolo import augment_yolo
from augmentations.augmentations_coco import augment_coco


# Default args.
INPUT_PATH = 'splitted_dataset'
OUTPUT_PATH = 'augmented_dataset'
FULL = False
TRAIN = False
TEST = False
VAL = False
COUNT = 5


def main():
    args = parse_args()

    subsets = []
    if args.full:
        subsets.append('full')
    if args.train:
        subsets.append('train')
    if args.test:
        subsets.append('test')
    if args.val:
        subsets.append('val')
    if not subsets:
        subsets.append('full')

    augment(format=format_name(args.input_path),
            input_path=args.input_path,
            output_path=args.output_path,
            subsets=subsets,
            count=args.count)


def augment(format: str, input_path: str, output_path: str, subsets: list, count: int) -> None:
    if format == 'yolo':
        augment_yolo(input_path, output_path, subsets, count)
    if format == 'voc':
        augment_voc(input_path, output_path, subsets, count)
    if format == 'coco':
        augment_coco(input_path, output_path, subsets, count)
    print("Output augmented dataset path:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Augment Yolo/MsCOCO/PascalVOc dataset or subsets.')
    parser.add_argument('--input', dest='input_path', type=str,
                        help="Path with dataset.", default=INPUT_PATH)
    parser.add_argument('--output', dest='output_path', type=str,
                        help="Path to save augmented dataset.", default=OUTPUT_PATH)
    parser.add_argument('--full', dest='full', action='store_true',
                        help='Augment full subset.', default=FULL)
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Augment train subset.', default=TRAIN)
    parser.add_argument('--test', dest='test', action='store_true',
                        help='Augment test subset.', default=TEST)
    parser.add_argument('--val', dest='val', action='store_true',
                        help='Augment val subset.', default=VAL)
    parser.add_argument('--count', dest='count', type=int,
                        help="Count of augmented copy of image.", default=COUNT)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
