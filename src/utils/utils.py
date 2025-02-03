import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


from src.configs import (
    IOU_PROPORTION,
    RESIZE_FACTOR,
    GREEN,
    CONTOUR_THICKNESS,
    MIN_LICENSE_PLATE_AREA_PERCENTAGE,
    MIN_OCCURENCES,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent


def generate_new_file_name(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = f"{base}_run{counter}{ext}"
    while os.path.exists(new_file_path):
        counter += 1
        new_file_path = f"{base}_run{counter}{ext}"
    return new_file_path


def save_recognized_plates(
    unique_license_plates, min_occurences=MIN_OCCURENCES
):
    recognized_license_plates_file = generate_new_file_name(
        "recognized_license_plates.json"
    )
    license_plates_frequency_file = generate_new_file_name(
        "license_plates_frequency.json"
    )

    frequent_plates = {
        plate: count
        for plate, count in unique_license_plates.items()
        if count >= min_occurences
    }
    license_plates_frequency = sorted(
        unique_license_plates.items(), key=lambda x: x[1], reverse=True
    )

    os.makedirs(BASE_DIR, exist_ok=True)
    recognized_license_plates_file = str(
        BASE_DIR / "results" / recognized_license_plates_file
    )
    license_plates_frequency_file = str(
        BASE_DIR / "results" / license_plates_frequency_file
    )
    json.dump(
        list(frequent_plates.items()), open(recognized_license_plates_file, "w")
    )
    json.dump(
        license_plates_frequency, open(license_plates_frequency_file, "w")
    )


def print_results(
    unique_license_plates, frame_counter, min_occurences=MIN_OCCURENCES
):
    frequent_plates = {
        plate: count
        for plate, count in unique_license_plates.items()
        if count >= min_occurences
    }
    if frame_counter % min_occurences == 0:
        print("Recognized License Plates:", list(frequent_plates.keys()))
        if frame_counter % (min_occurences * 10) == 0:
            print(
                "License Plates Counter:",
                sorted(
                    unique_license_plates.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ),
            )


def clean_license_plate(text):
    """
    Removes characters that don't normally appear in license plates.
    Keeps only alphanumeric characters and hyphens.
    """
    pattern = re.compile(r"[^A-Za-z0-9-]")
    cleaned_text = pattern.sub("", text)
    return cleaned_text


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
        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 3
    )
    return frame


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


def filter_detections_inside_area(
    results,
    analysis_area,
    frame_area,
    min_license_plate_area_percentage=MIN_LICENSE_PLATE_AREA_PERCENTAGE,
):
    if results is None:
        return None
    inside_area_indices = is_inside_predefined_area(
        results.boxes.xyxy.cpu().numpy(), analysis_area
    )
    x1, y1, x2, y2 = results.boxes.xyxy.cpu().numpy().T
    plate_area = (x2 - x1) * (y2 - y1)
    valid_indices = inside_area_indices & (
        plate_area / frame_area > min_license_plate_area_percentage
    )
    if np.any(valid_indices):
        return results[valid_indices]
    return None


def preprocess_for_ocr(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = apply_clahe_to_frame(image)

    # Binarization
    # _, image = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)

    # Morphological operations (dilation and erosion)
    # iterations = 1
    # for _ in range(iterations):
    #     kernel = np.ones((3, 3), np.uint8)
    #     image = cv2.dilate(image, kernel, iterations=1)
    #     image = cv2.erode(image, kernel, iterations=1)

    return image


def extract_text_from_bounding_boxes(
    frame, results, ocr_model, unique_license_plates
):
    texts = []
    if results is None:
        return texts
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
        cropped_image = frame[y1:y2, x1:x2]
        cropped_image = preprocess_for_ocr(cropped_image)
        text = ocr_model.run(cropped_image)
        if text:
            cleaned_text = clean_license_plate(text[0])
            if cleaned_text:
                texts.append(cleaned_text)
                unique_license_plates[cleaned_text] = (
                    unique_license_plates.get(cleaned_text, 0) + 1
                )
    return texts
