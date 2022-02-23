import os
import json
import random
import shutil
import argparse
import numpy as np
import pandas as pd

from distutils.dir_util import copy_tree

from config import VOC_NAME
from config import COCO_NAME
from config import YOLO_NAME
from utils import LABELS_ID
from utils import copy_all_images
from utils import make_empty_folder
from utils.format_specifier import format_name
from utils.format_specifier import is_splitted


# Default args.
INPUT_PATH = 'current_dataset'
OUTPUT_PATH = 'splitted_dataset'
TRAIN = 80
TEST = 15
VAL = 5
SEED = 42


def main():
    args = parse_args()

    random.seed(args.seed)

    split_dataset(input_path=args.input_path,
                  output_path=args.output_path,
                  proportions=[args.train, args.test, args.val],
                  seed=args.seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Split Yolo/MsCOCO/PascalVOc dataset on Train, Test, Val, TrainVal')
    parser.add_argument('--input', dest='input_path', type=str,
                        help="Path with dataset.", default=INPUT_PATH)
    parser.add_argument('--output', dest='output_path', type=str,
                        help="Path to save split dataset.", default=OUTPUT_PATH)
    parser.add_argument('--train', dest='train', type=int,
                        help="Train in in percents.", default=TRAIN)
    parser.add_argument('--test', dest='test', type=int,
                        help="Test in in percents.", default=TEST)
    parser.add_argument('--val', dest='val', type=int,
                        help="Val in in percents.", default=VAL)
    parser.add_argument('--seed', dest='seed', type=int,
                        help="Seed for random.", default=SEED)
    args = parser.parse_args()
    return args


def split_dataset(input_path: str, output_path: str, proportions: list, seed: int = 42) -> None:
    """
    Split dataset on train, test and val subsets.

    :param input_path: path to dataset.
    :param output_path: path to save splitted dataset.
    :param proportions: proportions of split on train, test and val.
    :param seed: seed of random.
    """
    assert is_splitted(input_path) is False, "Dataset should not be split!"

    print(f"Dataset on {input_path} is being split!")

    # Copy the dataset and split it.
    make_empty_folder(output_path)
    copy_tree(input_path, output_path)
    print("Output split dataset path:", output_path)

    # Split Yolo dataset.
    if format_name(output_path) == 'yolo':
        split_yolo(output_path, proportions, seed)

    # Split PascalVOC dataset.
    if format_name(output_path) == 'voc':
        split_voc(output_path, proportions, seed)

    # Split COCO dataset.
    if format_name(output_path) == 'coco':
        split_coco(output_path, proportions, seed)

    print("Dataset is splitted!")


def split_yolo(dataset_path: str, proportions: list, seed: int) -> None:
    """
    Split yolo dataset on train, test and val subsets.

    :param dataset_path: dataset path.
    :param proportions: proportions of split on train, test and val.
    :param seed: seed of random.
    """
    # Read all filenames.
    with open(f'{dataset_path}/{YOLO_NAME}/train_test_val.txt', 'r') as fr:
        images = fr.readlines()
    # Split on train, test and val subsets.
    train, test, val = split_list(images, proportions, seed)
    # Write on train.txt, test.txt and val.txt files.
    for subset, filename in zip([train, test, val], ['train.txt', 'test.txt', 'val.txt']):
        subset = change_path_yolo(subset, dataset_path)
        with open(f'{dataset_path}/{YOLO_NAME}/{filename}', 'w') as fw:
            fw.writelines(subset)
    # Write on trainval.txt file.
    for subset in [train, val]:
        with open(f'{dataset_path}/{YOLO_NAME}/trainval.txt', 'a') as fa:
            fa.writelines(subset)

    # Update dataset.data file.
    update_yolo_data(dataset_path)

    # Remove old file with all filenames.
    os.remove(f'{dataset_path}/{YOLO_NAME}/train_test_val.txt')


def split_voc(dataset_path: str, proportions: list, seed: int) -> None:
    """
    Split voc dataset on train, test and val subsets.

    :param dataset_path: dataset path.
    :param proportions: proportions of split on train, test and val.
    :param seed: seed of random.
    """
    # Read all base filenames.
    with open(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/train_test_val.txt', 'r') as fr:
        filenames = fr.readlines()
    # Split on train, test and val subsets.
    train, test, val = split_list(filenames, proportions, seed)
    # Write on train.txt, test.txt and val.txt files.
    for subset, filename in zip([train, test, val], ['train.txt', 'test.txt', 'val.txt']):
        with open(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/{filename}', 'w') as fw:
            fw.writelines(subset)
    # Write on trainval.txt file.
    for subset in [train, val]:
        with open(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/trainval.txt', 'a') as fa:
            fa.writelines(subset)
    # Remove old file with all filenames.
    os.remove(f'{dataset_path}/{VOC_NAME}/ImageSets/Main/train_test_val.txt')


def split_coco(dataset_path: str, proportions: list, seed: int) -> None:
    """
    Split voc dataset on train, test and val subsets.

    :param dataset_path: dataset path.
    :param proportions: proportions of split on train, test and val.
    :param seed: seed of random.
    """
    # Read data from coco json file.
    with open(f'{dataset_path}/{COCO_NAME}/Annotation/TrainTestVal.json', 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    # Split data and images (write on subsets directories).
    train, test, val = split_list(images, proportions, seed)
    split_coco_images(dataset_path, train, test, val)

    # Write on Train.json, Test.json, Val.json, TrainV
    save_coco_annotations(dataset_path, train, test, val, annotations, categories)


def change_path_yolo(paths: list, output_path: str) -> list:
    """
    Change dataset folder name in path to image.

    :param paths: list of paths.
    :param output_path: name of output folder.
    :return: list of changed paths.
    """
    def change(path: str) -> str:
        folders = path.split('/')
        folders[-4] = output_path
        return '/'.join(folders)

    return list(map(change, paths))


def split_list(data: list, proportions: list, seed: int) -> list:
    """
    Split list on `train`, `test` and `val` subsets with proportions.

    :param data: list with data.
    :param proportions: proportions of split on `train`, `test` and `val`.
    :param seed: seed of random.
    :return: list of subsets (`train`, `test`, `val`).
    """
    # We get the numbers of images data, by which we will then split.
    idx_all = np.arange(0, len(data))
    df = pd.DataFrame(idx_all)
    ids_train, ids_test, ids_val = np.split(df.sample(frac=1, random_state=seed),
                                            [int(proportions[0] / 100 * len(df)),
                                             int((100 - proportions[2]) / 100 * len(df))])
    ids_train = np.squeeze(ids_train.values)
    ids_val = np.squeeze(ids_val.values)
    ids_test = np.squeeze(ids_test.values)

    data = np.array(data)
    data_train = data[ids_train].tolist()
    data_test = data[ids_test].tolist()
    data_val = data[ids_val].tolist()

    return [data_train, data_test, data_val]


def update_yolo_data(dataset_path: str) -> None:
    """
    Update dataset.data file after split.

    :param dataset_path: path to dataset.
    """
    with open(f'{dataset_path}/{YOLO_NAME}/dataset.data', 'w') as wd:
        wd.write(f'classes = {len(LABELS_ID.keys())} \n')
        wd.write(f'train = {dataset_path}/{YOLO_NAME}/trainval.txt \n')
        wd.write(f'valid = {dataset_path}/{YOLO_NAME}/test.txt \n')
        wd.write(f'names = {dataset_path}/{YOLO_NAME}/dataset.names \n')
        wd.write('backup = path/to/backup/directory')


def split_coco_images(dataset_path: str, train: list, test: list, val: list) -> None:
    """
    Copy images from subtest to subset image folders.

    :param dataset_path: path to dataset.
    :param train: train subset.
    :param test: test subset.
    :param val: val subset.
    """
    # Image filenames sets.
    # train_filenames = list(map(lambda a: a['file_name'], train))
    # test_filenames = list(map(lambda a: a['file_name'], test))
    # val_filenames = list(map(lambda a: a['file_name'], val))
    train_filenames = [obj['file_name'] for obj in train]
    test_filenames = [obj['file_name'] for obj in test]
    val_filenames = [obj['file_name'] for obj in val]
    # Copy images from subtest to subset image folders.
    for subpath, image_set in zip(['Train', 'Test', 'Val'], [train_filenames, test_filenames, val_filenames]):
        from_path = f'{dataset_path}/{COCO_NAME}/TrainTestVal'
        to_path = f'{dataset_path}/{COCO_NAME}/{subpath}'
        copy_all_images(from_path, to_path, image_set)
        # Save on TrainVal folder.
        if subpath in ['Train', 'Val']:
            copy_all_images(from_path, f'{dataset_path}/{COCO_NAME}/TrainVal', image_set)
    # Delete old folder with all images.
    shutil.rmtree(f'{dataset_path}/{COCO_NAME}/TrainTestVal', ignore_errors=True)


def save_coco_annotations(dataset_path: str,
                          train: list, test: list, val: list,
                          annotations: list,
                          categories: list) -> None:
    """
    Save data to subsets json files.

    :param dataset_path: path to dataset.
    :param train: train subset.
    :param test: test subset.
    :param val: val subset.
    :param annotations: data of annotations.
    :param categories: data of categories.
    """
    # Save data to subsets (train, test, val) json files.
    for filename, images_subset in zip(['Train.json', 'Test.json', 'Val.json'], [train, test, val]):
        path = f'{dataset_path}/{COCO_NAME}/Annotation/{filename}'
        subset_annotations = filter_coco_annotations(annotations, images_subset)
        save_coco_annotation(path, images_subset, subset_annotations, categories)

    # Write TrainVal.json file.
    trainval_ann = filter_coco_annotations(annotations, train) + \
                   filter_coco_annotations(annotations, val)
    trainval_images = train + val
    trainval_path = f'{dataset_path}/{COCO_NAME}/Annotation/TrainVal.json'
    save_coco_annotation(trainval_path, trainval_images, trainval_ann, categories)
    # Delete old json file.
    os.remove(f'{dataset_path}/{COCO_NAME}/Annotation/TrainTestVal.json')


def filter_coco_annotations(annotations: list, images: list) -> list:
    """
    Find annotations only from current subset.

    :param annotations: all annotations data.
    :param images: subset with images data.
    :return: list subset annotations.
    """
    # image_ids = list(map(lambda i: int(i['id']), images))
    # return list(filter(lambda a: int(a['image_id']) in image_ids, annotations))
    image_ids = [img_data['id'] for img_data in images]
    result = [ann for ann in annotations if ann['image_id'] in image_ids]
    return result


def save_coco_annotation(path: str, images: list, annotations: list, categories: list) -> None:
    """
    Save coco data to json file.

    :param path: path to save.
    :param images: images data.
    :param annotations: annotations data.
    :param categories: categories data.
    """
    with open(path, 'wt', encoding='UTF-8') as coco:
        json.dump({'images': images,
                   'categories': categories,
                   'annotations': annotations}, coco, indent=2)


if __name__ == '__main__':
    main()
