from time import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer

from utils.utils import (
    adaptive_resize,
    apply_clahe_to_frame,
    filter_detections_inside_area,
    draw_area,
    draw_text,
    print_results,
    save_recognized_plates,
    extract_text_from_bounding_boxes,
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
    VEHICLE_CLASS_IDS,
)

MIN_LICENSE_PLATE_AREA_PERCENTAGE = (
    0.01  # Minimum license plate area percentage of the input image area
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


def postprocess_license_plate(frame, results, analysis_area):
    frame_area = frame.shape[0] * frame.shape[1]
    results = results[results.boxes.conf > CONF_TH] if results else None
    results = filter_detections_inside_area(
        results, analysis_area, frame_area, MIN_LICENSE_PLATE_AREA_PERCENTAGE
    )
    return extract_text_from_bounding_boxes(
        frame, results, ocr_model, unique_license_plates
    )


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
            if not ret or frame_counter > 60:
                break

            frame = process_frame(frame, analysis_area)
            print_results(unique_license_plates, frame_counter)
            cv2.imshow("License Plate Recognition", frame)

            if SAVE_VIDEO:
                s.write_frame(frame=frame)

            frame_counter += 1
            total_proc_time = int((time() - start) * 1000)
            if cv2.waitKey(max(1, 33 - total_proc_time)) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    save_recognized_plates(unique_license_plates)


if __name__ == "__main__":
    main()
