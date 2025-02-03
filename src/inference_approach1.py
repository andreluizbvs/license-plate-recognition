import json
import os
from time import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer

from utils.utils import (
    clean_license_plate,
    adaptive_resize,
    apply_clahe_to_frame,
    is_inside_predefined_area,
    draw_area,
    draw_text,
    generate_new_file_name,
)
from configs import (
    CONTOUR_THICKNESS,
    SAVE_VIDEO,
    VEHICLE_WEIGHTS_PATH,
    LICENSE_PLATE_WEIGHTS_PATH,
    VIDEO_PATH,
    CONF_TH,
    HEIGHT_PART_ANALYSYS,
    MIN_CAR_AREA_PERCENTAGE,
    MIN_LICENSE_PLATE_AREA_PERCENTAGE,
    MIN_OCCURENCES,
    VEHICLE_CLASS_IDS,
)

# Global instances
vehicle_model = YOLO(VEHICLE_WEIGHTS_PATH)  # Detect vehicles
license_plate_model = YOLO(
    LICENSE_PLATE_WEIGHTS_PATH
)  # Detect license plates in vehicles bboxes
ocr_model = ONNXPlateRecognizer(
    "global-plates-mobile-vit-v2-model", device="cpu"
)  # Recognize license plates

# Results
unique_license_plates = {}


def filter_detections_inside_area(results, analysis_area, frame_area):
    if results is None:
        return None
    inside_area_indices = is_inside_predefined_area(
        results.boxes.xyxy.cpu().numpy(), analysis_area
    )
    x1, y1, x2, y2 = results.boxes.xyxy.cpu().numpy().T
    object_area = (x2 - x1) * (y2 - y1)
    valid_indices = inside_area_indices & (
        object_area / frame_area < MIN_LICENSE_PLATE_AREA_PERCENTAGE
    )
    if np.any(valid_indices):
        return results[valid_indices]
    return None


def preprocess_for_ocr(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = apply_clahe_to_frame(image)
    return image


def extract_text_from_bounding_boxes(frame, results):
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


def postprocess_license_plate(frame, results, analysis_area):
    frame_area = frame.shape[0] * frame.shape[1]
    results = results[results.boxes.conf > CONF_TH] if results else None
    results = filter_detections_inside_area(results, analysis_area, frame_area)
    texts = extract_text_from_bounding_boxes(frame, results)
    return texts


def preprocess(frame):
    return apply_clahe_to_frame(frame)


def predict(frame, analysis_area):
    vehicle_results = vehicle_model.track(frame, verbose=False)[0]
    vehicle_boxes = vehicle_results.boxes.cpu().numpy()
    vehicle_boxes = vehicle_boxes[np.isin(vehicle_boxes.cls, VEHICLE_CLASS_IDS)]

    license_plate_crops = []
    for vehicle_box in vehicle_boxes:
        x1, y1, x2, y2 = map(int, vehicle_box.xyxy.flatten())
        vehicle_crop = frame[y1:y2, x1:x2]

        car_area_proportion = (vehicle_crop.size) / frame.size
        if (
            vehicle_crop.size == 0
            or vehicle_crop.shape[0] == 0
            or vehicle_crop.shape[1] == 0
            or car_area_proportion < MIN_CAR_AREA_PERCENTAGE
        ):
            continue

        # Run license plate model on each extracted vehicle
        license_plate_results = license_plate_model(
            vehicle_crop, verbose=False
        )[0]
        texts = postprocess_license_plate(
            vehicle_crop, license_plate_results, analysis_area
        )
        for license_plate_box in license_plate_results.boxes:
            x1_lp, y1_lp, x2_lp, y2_lp = map(
                int, license_plate_box.xyxy.flatten()
            )
            license_plate_crops.append(
                [x1 + x1_lp, y1 + y1_lp, x1 + x2_lp, y1 + y2_lp, texts]
            )
    return license_plate_crops


def postprocess(frame, license_plate_crops):
    for license_plate_crop in license_plate_crops:
        x1, y1, x2, y2, texts = license_plate_crop
        draw_area(frame, [x1, y1, x2, y2])
        if texts:
            draw_text(frame, f"{texts[0]}", x1, y1)
    return adaptive_resize(frame)


def process_frame(frame, analysis_area):
    frame = preprocess(frame)
    license_plate_crops = predict(frame, analysis_area)
    return postprocess(frame, license_plate_crops)


def save_recognized_plates():
    recognized_license_plates_file = generate_new_file_name(
        "recognized_license_plates.json"
    )
    license_plates_frequency_file = generate_new_file_name(
        "license_plates_frequency.json"
    )

    frequent_plates = {
        plate: count
        for plate, count in unique_license_plates.items()
        if count >= MIN_OCCURENCES
    }
    license_plates_frequency = sorted(
        unique_license_plates.items(), key=lambda x: x[1], reverse=True
    )
    json.dump(
        list(frequent_plates.items()), open(recognized_license_plates_file, "w")
    )
    json.dump(
        license_plates_frequency, open(license_plates_frequency_file, "w")
    )


def print_results(frame_counter):
    frequent_plates = {
        plate: count
        for plate, count in unique_license_plates.items()
        if count >= MIN_OCCURENCES
    }
    if frame_counter % MIN_OCCURENCES == 0:
        print("Recognized License Plates:", list(frequent_plates.keys()))
        if frame_counter % (MIN_OCCURENCES * 10) == 0:
            print(
                "License Plates Counter:",
                sorted(
                    unique_license_plates.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ),
            )


def main():
    video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    analysis_area = (
        ((CONTOUR_THICKNESS // 2) + 1),
        h // HEIGHT_PART_ANALYSYS,
        w - ((CONTOUR_THICKNESS // 2) + 1),
        (HEIGHT_PART_ANALYSYS - 1) * h // HEIGHT_PART_ANALYSYS,
    )

    print(f"Original video res. (width x height): {w}x{h}")
    print("Original video FPS: ", cap.get(cv2.CAP_PROP_FPS))
    print(
        "Original video duration: ",
        cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        "seconds",
    )

    frame_counter = 0

    with sv.VideoSink(
        target_path=VIDEO_PATH.replace(".mp4", "_labeled.mp4"),
        video_info=video_info,
    ) as s:
        while True:
            start = time()

            ret, frame = cap.read()
            if not ret or frame_counter > 20:
                break

            frame = process_frame(frame, analysis_area)
            print_results(frame_counter)
            cv2.imshow("License Plate Recognition", frame)

            if SAVE_VIDEO:
                s.write_frame(frame=frame)

            frame_counter += 1
            total_proc_time = int((time() - start) * 1000)
            if cv2.waitKey(max(1, 33 - total_proc_time)) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    save_recognized_plates()


if __name__ == "__main__":
    main()
