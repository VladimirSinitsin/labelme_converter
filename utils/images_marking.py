import cv2
import random
import numpy as np

from tqdm import tqdm
from typing import Tuple

from utils import ID2NAMES
from utils import LABELS_ID
from utils import make_empty_folder


COLORS = {0: (0, 0, 128),
          1: (0, 128, 0),
          2: (0, 128, 128),
          3: (128, 0, 0),
          4: (128, 0, 128),
          5: (128, 128, 0),
          6: (128, 128, 128),
          7: (0, 0, 64)}


def create_marked_images(input_path: str, output_path: str, labelme_data: list) -> None:
    """
    Create marked images from dataset.

    :param input_path: path with original images.
    :param output_path: path to save marked images.
    :param labelme_data: data of objects on images.
    """
    # Make outpath.
    make_empty_folder(f"{output_path}/MarkedImages")
    print('Create marked images:')
    for image_data in tqdm(labelme_data):
        image_path = f"{input_path}/{image_data['filename']}"
        image = cv2.imread(image_path)
        marked_image = mark_image(image, image_data['objects'])
        cv2.imwrite(f"{output_path}/MarkedImages/marked_{image_data['filename']}", marked_image)


def mark_image(image: np.ndarray, objects: list) -> np.ndarray:
    """
    Draw objects on image.

    :param image: source image.
    :param objects: list of objects.
    :return: image with objects.
    """
    h, w, d = image.shape
    # Image with translucent fills.
    added_img = np.zeros([h, w, d], dtype=np.uint8)
    screen_h = 1080
    screen_w = 1920
    # Thickness of lines.
    scale = np.min([float(screen_h) / float(h), float(screen_w) / float(w)])
    for obj in objects:
        image = draw_object(obj, image, added_img, scale)
    # Overlay polygon fills.
    cv2.addWeighted(added_img, 0.8, image, 0.9, 0, image)
    return image


def draw_object(obj: dict, src_img: np.ndarray, added_img: np.ndarray, scale: np.float) -> np.ndarray:
    """
    Draw labeled object on image.

    :param obj: dictionary with data about object.
    :param src_img: image to draw edging.
    :param added_img: image to draw fills.
    :param scale: scale of thickness edging.
    :return: marked image.
    """
    class_name = obj['name']
    color = get_color(LABELS_ID[class_name])
    coords = convert_coords(obj['points']) if 'poly_points' not in obj.keys() else obj['poly_points']
    # Draw the fill.
    cv2.fillConvexPoly(added_img, coords, color)
    # Draw rectangle.
    cv2.polylines(src_img, [coords], True, color, thickness=int(4.0 / scale))
    # Draw text.
    x1, y1 = coords[0]
    src_img = draw_text_box(src_img, class_name, x1, y1)
    return src_img


def get_color(id: int) -> Tuple[int, int, int]:
    """
    Return the color of object.

    :param id: id of object.
    :return: color.
    """
    if id in COLORS.keys():
        return COLORS[id]
    new_color = COLORS[0]
    colors_values = COLORS.values()
    while new_color in colors_values:
        new_color = (random.randint(20, 230), random.randint(20, 230), random.randint(20, 230))
    COLORS[len(COLORS)] = new_color
    return new_color


def convert_coords(coords: list) -> np.ndarray:
    """
    Convert coords from (x_min, y_min, x_max, y_max) to ([x1, y1], [x2, y2], [x3, y3], [x4, y4]).

    :param coords: coords of points.
    :return: converted coords.
    """
    x_min, y_min, x_max, y_max = coords
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
    xy = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype(int)  # Polygon
    return xy


def draw_text_box(img: np.array, text: str, x: int, y: int,
                  font_color=(255, 255, 255), back_color=(0, 0, 0), font_scale=0.5, thickness=1) -> np.ndarray:
    """
    Draw small text box on image.

    :param img: image to draw.
    :param text: text to draw.
    :param x: coord x.
    :param y: coord y.
    :param font_color: color of font.
    :param back_color: color of background.
    :param font_scale: scale of font.
    :param thickness: thickness of text.
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Get the width and height of the text box.
    t_w, t_h = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    # Make the coords of the box with a small padding of two pixels.
    box_coords = [(int(x), int(y + 5)), (int(x + t_w), int(y - t_h))]
    cv2.rectangle(img, box_coords[0], box_coords[1], back_color, cv2.FILLED)
    cv2.putText(img, str(text), (int(x+1), int(y+1)), font, fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    cv2.putText(img, str(text), (int(x), int(y)), font, fontScale=font_scale, color=font_color, thickness=thickness)
    return img


def create_mark_objects(bboxes: list, categories_idx: list) -> list:
    """
    Create list with objects data.

    :param bboxes: bboxes of objects.
    :param categories_idx: idx of categories.
    :return: list with objects data.
    """
    categories = [ID2NAMES[cat_id] for cat_id in categories_idx]
    objects = []
    for cat, bbox in zip(categories, bboxes):
        obj = {'name': cat,
               'points': bbox}
        objects.append(obj)
    return objects
