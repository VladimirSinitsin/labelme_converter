import lxml.etree as ET

from tqdm import tqdm
from typing import Tuple

from converters import Converter
from config import VOC_NAME
from utils import make_empty_folder, copy_all_images


class Converter2voc(Converter):
    def convert(self) -> None:
        """ Convert labelme dataset to PascalVOC format. """
        make_empty_folder(f"{self.output_path}/{VOC_NAME}")

        # Make subdirectories: `/Annotations`, `/JPEGImages`, `/ImageSets/Main`.
        self._make_dataset_dirs()

        # Write images.
        copy_all_images(self.input_path, f"{self.output_path}/{VOC_NAME}/JPEGImages")

        print('Converting labelme to PascalVOC:')
        for image_data in tqdm(self.data):
            self._convert_image_data(image_data)

    def _make_dataset_dirs(self) -> None:
        """ Make service directories. """
        make_empty_folder(f"{self.output_path}/{VOC_NAME}/Annotations")
        make_empty_folder(f"{self.output_path}/{VOC_NAME}/JPEGImages")
        make_empty_folder(f"{self.output_path}/{VOC_NAME}/ImageSets/Main")

    def _convert_image_data(self, image_data: dict) -> None:
        """
        Convert labelme data of current image to VOC format.

        :param image_data: labelme data of image.
        """
        hwd = image_data['height'], image_data['width'], image_data['depth']
        # Convert and create annotation file.
        tree = self.create_file(image_data['filename'], hwd, image_data['objects'])
        # Write annotation file.
        xml_path = f"{self.output_path}/{VOC_NAME}/Annotations/{image_data['filename'].split('.')[0]}.xml"
        with open(xml_path, 'wb') as f:
            # f.write('<?xml version="1.0" encoding="utf-8"?>\n'.encode())
            f.write(ET.tostring(tree, pretty_print=True))
        # Write base image names in txt set.
        txt_file_path = f"{self.output_path}/{VOC_NAME}/ImageSets/Main/train_test_val.txt"
        with open(txt_file_path, 'a') as ft:
            ft.write(image_data['filename'].split('.')[0] + '\n')

    @staticmethod
    def create_file(filename: str, hwd: Tuple[int, int, int], img_objects: list) -> ET.ElementTree:
        """
        Create file in VOC format.

        :param filename: image file name.
        :param hwd: height, width and depth of image.
        :param img_objects: objects on image annotations in labelme format.
        :return: file for write in xml file.
        """
        root = Converter2voc._create_root(filename, hwd[0], hwd[1], hwd[2])
        root = Converter2voc._create_object_annotation(root, img_objects)
        tree = ET.ElementTree(root)
        return tree

    @staticmethod
    def _create_root(filename: str, height: int, width: int, depth: int) -> ET.Element:
        """
        Create root info in xml file.

        :param filename: image file name.
        :param width: width of image.
        :param height: height of image.
        :param depth: depth of image.
        :return: xml file with root info.
        """
        root = ET.Element("annotations")
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "folder").text = "JPEGImages"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = f'{width}'
        ET.SubElement(size, "height").text = f'{height}'
        ET.SubElement(size, "depth").text = f'{depth}'
        return root

    @staticmethod
    def _create_object_annotation(root: ET.Element, img_objects: list) -> ET.Element:
        """
        Create objects annotation in xml file.

        :param root: xml file with root info.
        :param img_objects: data of objects on image in labelme format.
        :return: xml file with objects annotation.
        """
        for img_object in img_objects:
            x_min, y_min, x_max, y_max = img_object['points']
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = img_object['name']
            ET.SubElement(obj, "pose").text = "Frontal"
            ET.SubElement(obj, "truncated").text = '0'
            ET.SubElement(obj, "difficult").text = '0'
            ET.SubElement(obj, "content").text = '###'
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = f'{x_min}'
            ET.SubElement(bbox, "ymin").text = f'{y_min}'
            ET.SubElement(bbox, "xmax").text = f'{x_max}'
            ET.SubElement(bbox, "ymax").text = f'{y_max}'
        return root
