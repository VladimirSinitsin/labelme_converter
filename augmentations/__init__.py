import numpy as np
import albumentations as A


def augmentation_image(format: str, image: np.ndarray, bboxes: list, classes_ids: list) -> dict:
    # Declare an augmentation pipeline
    aug_chain = A.Compose([
        A.GaussNoise(var_limit=30, p=0.5),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Perspective(scale=0.03, keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, interpolation=1, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=3, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=5,
                         num_flare_circles_upper=20, src_radius=20, src_color=(255, 255, 255), p=0.5),
        ], A.BboxParams(format=format, label_fields=['classes_ids']))

    transformed = aug_chain(image=image, bboxes=bboxes, classes_ids=classes_ids)
    return transformed
