import json

import numpy as np

from tqdm import tqdm

from converters import Converter
from config import COCO_NAME
from utils import LABELS_ID
from utils import make_empty_folder, copy_all_images


class Converter2coco(Converter):
    ann_id = 1

    def convert(self) -> None:
        """ Convert labelme dataset to MsCOCO format. """
        make_empty_folder(f"{self.output_path}/{COCO_NAME}")

        # Copy all images
        images_path = f"{self.output_path}/{COCO_NAME}/TrainTestVal"
        make_empty_folder(images_path)
        copy_all_images(self.input_path, images_path)

        # Convert labelme to coco format and create JSON annotation.
        coco_format = self._create_coco_format(self.data)
        # Write JSON file.
        make_empty_folder(f"{self.output_path}/{COCO_NAME}/Annotation")
        with open(f"{self.output_path}/{COCO_NAME}/Annotation/TrainTestVal.json", 'w') as outfile:
            json.dump(coco_format, outfile, indent=2)

    def _create_coco_format(self, labelme_data: np.ndarray) -> dict:
        """
        Create coco format annotation.

        :param labelme_data: data in labelme format.
        :return: dictionary with coco format annotation.
        """
        coco_format = {'images': [],
                       'categories': self._create_categories(),
                       'annotations': []}
        # Create `coco_format['images']` and `coco_format['annotations']`.
        print('Converting labelme to MsCOCO:')
        for image_id, image_data in enumerate(tqdm(labelme_data)):
            image = {'file_name': image_data['filename'],
                     'height': image_data['height'],
                     'width': image_data['width'],
                     'id': image_id * 100}
            coco_format['images'].append(image)
            coco_format['annotations'].extend(self._create_annotations(image_data['objects'], image_id * 100))
        return coco_format

    @staticmethod
    def _create_categories() -> list:
        """
        Create `categories` field in coco annotation.

        :return: list with categories in coco format annotation.
        """
        categories = []
        for label, id in LABELS_ID.items():
            ann = {"supercategory": "supercategory",
                   "id": id + 1,  # Index starts with '1'
                   "name": label}
            categories.append(ann)
        return categories

    @staticmethod
    def _create_annotations(objects_data: list, image_id: int) -> list:
        """
        Create annotation of objects on current image in coco format.

        :param objects_data: data of objects on image.
        :param image_id: id og image.
        :return: list with objects annotation in coco format.
        """
        annotations = []
        for obj_data in objects_data:
            x_min, y_min, x_max, y_max = obj_data['points']
            bbox = [int(item) for item in [x_min, y_min, x_max - x_min, y_max - y_min]]
            area = (x_max - x_min) * (y_max - y_min)  # width * height
            try:
                category_id = LABELS_ID[obj_data['name']] + 1
            except:
                print(f"Object in image with id={image_id}: {obj_data['name']} not found!")
            else:
                annotation = {'id': Converter2coco.ann_id * 100,
                              'image_id': image_id,
                              'bbox': bbox,
                              'area': int(area),
                              'iscrowd': 0,
                              'category_id': category_id,
                              'segmentation': []}
                annotations.append(annotation)
                Converter2coco.ann_id += 1
        return annotations
