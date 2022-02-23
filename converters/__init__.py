import os
import abc
import cv2
import json
import numpy as np

from tqdm import tqdm

from utils import make_empty_folder
from utils.images_marking import create_marked_images


class Converter:
    def __init__(self, input_path: str, output_path: str, create_marked: bool, polygons: bool) -> None:
        """
        Create converter from labelme format.

        :param input_path: path to labelme dataset.
        :param output_path: path to save converted dataset.
        :param create_marked: create marked images.
        """
        self.input_path = input_path
        self.output_path = output_path
        make_empty_folder(self.output_path)

        print('Input path: ', self.input_path)
        print('Output path: ', self.output_path)

        # Draw marked images with polygons (not rectangles).
        self.polygons = polygons

        # Reading data from labelme dataset.
        self.data = self._get_data()

        # Create and save marked images.
        if create_marked:
            create_marked_images(self.input_path, self.output_path, self.data)

    def _get_data(self) -> list:
        """
        Read and reformat data for labelme dataset.

        :return: data.

        Returned data format:

        data (list):[
           image0_data (dict):{
                'filename': image0_filename.jpg,
                'height': 1080,
                'width': 1920,
                'depth': 3,
                'objects' (list):[
                    object_0 (dict):{
                        'name': 'class_name',
                        'points' (list): [x_min, y_min, x_max, y_max]
                    },
                    object_1 (dict):{
                        ...
                    },
                    ...
                ]
            },
            image1_data (dict):{
                ...
            },
            ...
        ]
        """
        json_list = []
        files = os.listdir(self.input_path)
        for file in files:
            if file.endswith(".json"):
                json_list.append(self.input_path + '/' + file)

        data = []
        print('Reading labelme files:')
        for json_path in tqdm(json_list):
            with open(json_path, 'r') as fp:
                json_data = json.load(fp)
            image_data = {}
            image_data['filename'] = json_data['imagePath']
            image_data['height'] = json_data['imageHeight']
            image_data['width'] = json_data['imageWidth']
            image_data['depth'] = self._get_img_depth(f"{self.input_path}/{image_data['filename']}")
            image_data['objects'] = []
            for shape in json_data['shapes']:
                object = {}
                object['name'] = shape['label']
                if self.polygons:
                    object['poly_points'] = self._reform_polygon(points=shape['points'],
                                                                 img_width=image_data['width'],
                                                                 img_height=image_data['height'])
                # [x_min, y_min, x_max, y_max]
                object['points'] = self._get_coords(points=shape['points'],
                                                    img_width=image_data['width'],
                                                    img_height=image_data['height'])
                image_data['objects'].append(object)
            data.append(image_data)

        return data

    @staticmethod
    def _get_img_depth(img_path: str) -> int:
        """
        Determine the number of channels in the image.

        :return: number of channels.
        """
        img = cv2.imread(img_path)
        return img.shape[2]

    @staticmethod
    def _reform_polygon(points: list, img_width: int, img_height: int) -> np.ndarray:
        """ Round and clip coords. """
        points = np.array(points).round().astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, img_width)
        points[:, 1] = np.clip(points[:, 1], 0, img_height)
        return points

    @staticmethod
    def _get_coords(points: list, img_width: int, img_height: int) -> list:
        """
        Convert points from (x1, y1, x2, y2, x3, y3, x4, y4) to (x_min, y_min, x_max, y_max).

        :param points: coordinate of polygon.
        :return: coordinate of rectangle.
        """
        points = np.array(points).astype(int)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        x_min, x_max = np.clip([x_min, x_max], 0, img_width)
        y_min, y_max = np.clip([y_min, y_max], 0, img_height)
        return [x_min, y_min, x_max, y_max]

    @abc.abstractmethod
    def convert(self) -> None:
        """ Convert to selected dataset type. """
