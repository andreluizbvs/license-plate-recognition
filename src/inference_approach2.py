import json
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
    LICENSE_PLATE_WEIGHTS_PATH,
    VIDEO_PATH,
    CONF_TH,
    HEIGHT_PART_ANALYSYS,
    MIN_OCCURENCES,
)

MIN_LICENSE_PLATE_AREA_PERCENTAGE = (
    0.00125  # Minimum license plate area percentage of the input image area
)

# Global instances
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
    valid_indices = inside_area_indices
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
        plate_area = (x2 - x1) * (y2 - y1)
        if plate_area < MIN_LICENSE_PLATE_AREA_PERCENTAGE * frame_area:
            valid_indices[i] = False

    if np.any(valid_indices):
        return results[valid_indices]
    return None


def annotate_frame(frame, results, texts):
    if results is None:
        return frame
    for box, text in zip(results.boxes, texts):
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
        conf = float(box.conf.cpu().numpy())
        cls = int(box.cls.cpu().numpy())
        label = (
            f"{text} {conf:.2f}"
            if text
            else f"{license_plate_model.names[cls]} {conf:.2f}"
        )
        draw_area(frame, (x1, y1, x2, y2))
        draw_text(frame, label, x1, y1)
    return frame


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


def extract_text_from_bounding_boxes(frame, results):
    texts = []
    if results is None:
        return texts
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
        cropped_image = frame[y1:y2, x1:x2]
        cv2.imshow("License Plate Recognition", cropped_image)
        cropped_image = preprocess_for_ocr(cropped_image)
        text = ocr_model.run(cropped_image)
        if text:
            cleaned_text = clean_license_plate(text[0])
            texts.append(cleaned_text)
            if cleaned_text:
                unique_license_plates[cleaned_text] = (
                    unique_license_plates.get(cleaned_text, 0) + 1
                )
    return texts


def preprocess(frame):
    return apply_clahe_to_frame(frame)


def predict(frame):
    return license_plate_model.track(frame, persist=True, verbose=False)[0]


def postprocess(frame, results, analysis_area):
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width

    results = results[results.boxes.conf > CONF_TH] if results else None
    results = filter_detections_inside_area(results, analysis_area, frame_area)
    texts = extract_text_from_bounding_boxes(frame, results)
    frame = annotate_frame(frame, results, texts)

    draw_area(frame, analysis_area)
    return adaptive_resize(frame)


def process_frame(frame, analysis_area):
    frame = preprocess(frame)
    results = predict(frame)
    return postprocess(frame, results, analysis_area)


def save_recognized_plates():
    recognized_license_plates_file = generate_new_file_name("recognized_license_plates.json")
    license_plates_frequency_file = generate_new_file_name("license_plates_frequency.json")

    frequent_plates = {
        plate: count
        for plate, count in unique_license_plates.items()
        if count >= MIN_OCCURENCES
    }
    license_plates_frequency = sorted(unique_license_plates.items(), key=lambda x: x[1], reverse=True)
    json.dump(list(frequent_plates.items()), open(recognized_license_plates_file, "w"))
    json.dump(license_plates_frequency, open(license_plates_frequency_file, "w"))


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
