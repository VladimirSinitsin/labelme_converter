import cv2

from typing import Tuple


def resize_image(img_path: str, new_w: int, new_h: int) -> None:
    """
    Resize and write image.

    :param img_path: path to image.
    :param new_w: new width.
    :param new_h: new height.
    """
    image = cv2.imread(img_path)
    # If it is not a marked image and it is not found, raise an exception.
    # Marked image may not have been created during converting, so skip this path to image.
    if image is None and 'MarkedImages' not in img_path:
        raise Exception(f"Image on {img_path} was not found!")
    elif image is not None:
        new_image = cv2.resize(image, (new_w, new_h), cv2.INTER_NEAREST)
        cv2.imwrite(img_path, new_image)


def get_ratio(img_path: str, new_w: int, new_h: int) -> Tuple[float, float]:
    """
    Get ratio of new and old shape.

    :param img_path: path to image.
    :param new_w: new width.
    :param new_h: new height.
    :return: ratio width, ratio height.
    """
    image = cv2.imread(img_path)
    img_h, img_w, _ = image.shape
    ratio_w = new_w / img_w
    ratio_h = new_h / img_h
    return ratio_w, ratio_h
