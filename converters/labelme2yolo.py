from tqdm import tqdm

from converters import Converter
from config import YOLO_NAME
from utils import to_fixed, make_empty_folder, copy_all_images
from utils import LABELS_ID
from utils import SCRIPT_PATH


class Converter2yolo(Converter):
    def convert(self) -> None:
        """ Convert labelme dataset to Yolo format. """
        make_empty_folder(f"{self.output_path}/{YOLO_NAME}")
        # Create `dataset.names` and `dataset.data` files.
        self._create_info_files(self.output_path)

        ann_path = f"{self.output_path}/{YOLO_NAME}/dataset_data"
        make_empty_folder(ann_path)
        copy_all_images(self.input_path, ann_path)

        print('Converting labelme to Yolo:')
        for image_data in tqdm(self.data):
            self._convert_image_data(image_data, ann_path)
            self._write_set_row(image_data['filename'])

    @staticmethod
    def _create_info_files(output_path: str) -> None:
        """
        Create info files `dataset.names` and `dataset.data`.

        :param output_path: path to save files.
        """
        with open(f"{output_path}/{YOLO_NAME}/dataset.names", 'w') as wn:
            for label in LABELS_ID.keys():
                wn.write(label + '\n')
        with open(f"{output_path}/{YOLO_NAME}/dataset.data", 'w') as wd:
            wd.write(f"classes = {len(LABELS_ID.keys())} \n")
            wd.write(f"train = {output_path}/dataset_train_test_val.txt \n")
            wd.write('backup = path/to/backup/directory')

    def _convert_image_data(self, image_data: dict, ann_path: str) -> None:
        """
        Convert data of image in Yolo format.

        :param image_data: data of current image.
        :param ann_path: path to save annotations.
        """
        txt_filename = f"/{image_data['filename'].split('.')[0]}.txt"
        with open(ann_path + txt_filename, 'a') as wt:
            for object in image_data['objects']:
                # [x, y, w, h]
                coords = self.convert_coords(object['points'], image_data['width'], image_data['height'])
                try:
                    row = f"{LABELS_ID[object['name']]} {coords[0]} {coords[1]} {coords[2]} {coords[3]}"
                    wt.write(row + '\n')
                except:
                    print(f"Object in {image_data['filename']} {object['name']} not found!")

    @staticmethod
    def convert_coords(points: list, img_width: int, img_height: int) -> list:
        """
        Convert coords from (x_min, y_min, x_max, y_max) to (x, y, width, height).
        `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.

        :param points: lis of points in (x_min, y_min, x_max, y_max) format.
        :param img_width: width of image.
        :param img_height: height of image.
        :return: lis of points in (x, y, width, height) format.
        """
        x_min, y_min, x_max, y_max = points
        object_width = x_max - x_min
        object_height = y_max - y_min
        x = ((x_min + x_max) / 2) / img_width
        y = ((y_min + y_max) / 2) / img_height
        w = object_width / img_width
        h = object_height / img_height
        return list(map(to_fixed, [x, y, w, h]))

    def _write_set_row(self, image_filename: str):
        with open(f"{self.output_path}/{YOLO_NAME}/train_test_val.txt", 'a') as wt:
            wt.write(f"{SCRIPT_PATH}/{self.output_path}/{YOLO_NAME}/dataset_data/{image_filename}\n")
