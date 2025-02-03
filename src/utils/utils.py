import os
import re

import cv2
import numpy as np

from configs import (
    IOU_PROPORTION,
    RESIZE_FACTOR,
    GREEN,
    CONTOUR_THICKNESS,
)


def clean_license_plate(text):
    """
    Removes characters that don't normally appear in license plates.
    Keeps only alphanumeric characters and hyphens.
    """
    pattern = re.compile(r"[^A-Za-z0-9-]")
    cleaned_text = pattern.sub("", text)
    return cleaned_text


def adaptive_resize(frame, max_height=720, max_width=1280):
    if frame is None:
        return None
    height, width = frame.shape[:2]
    while height > max_height or width > max_width:
        frame = cv2.resize(
            frame, (width // RESIZE_FACTOR, height // RESIZE_FACTOR)
        )
        height, width = frame.shape[:2]
    return frame


def apply_clahe_to_frame(frame):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return cv2.merge(tuple(map(clahe.apply, cv2.split(frame))))


def is_inside_predefined_area(xyxy, area):
    x_min, y_min, x_max, y_max = area
    x1, y1, x2, y2 = xyxy.T
    object_area = (x2 - x1) * (y2 - y1)
    inter_x1 = np.maximum(x1, x_min)
    inter_y1 = np.maximum(y1, y_min)
    inter_x2 = np.minimum(x2, x_max)
    inter_y2 = np.minimum(y2, y_max)
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(
        0, inter_y2 - inter_y1
    )
    return inter_area >= IOU_PROPORTION * object_area


def draw_area(frame, area):
    if area[0] > area[2] or area[1] > area[3]:
        return frame
    cv2.rectangle(
        frame,
        (area[0], area[1]),
        (area[2], area[3]),
        GREEN,
        CONTOUR_THICKNESS,
    )
    return frame


def draw_text(frame, label, x1, y1):
    cv2.putText(
        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, GREEN, 6
    )
    return frame


def generate_new_file_name(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = f"{base}_run{counter}{ext}"
    while os.path.exists(new_file_path):
        counter += 1
        new_file_path = f"{base}_run{counter}{ext}"
    return new_file_path

