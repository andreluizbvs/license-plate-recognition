from time import time

import cv2
import supervision as sv
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer

from src.utils.utils import (
    adaptive_resize,
    apply_clahe_to_frame,
    filter_detections_inside_area,
    draw_area,
    draw_text,
    print_results,
    save_recognized_plates,
    extract_text_from_bounding_boxes,
)

from src.configs import (
    CONTOUR_THICKNESS,
    SAVE_VIDEO,
    LICENSE_PLATE_WEIGHTS_PATH,
    CONF_TH,
    HEIGHT_PART_ANALYSYS,
    MIN_OCCURENCES
)

MIN_LICENSE_PLATE_AREA_PERCENTAGE = (
    0.00125  # Minimum license plate area percentage of the input image area
)

# Global instances
license_plate_model = YOLO(
    LICENSE_PLATE_WEIGHTS_PATH
)  # Detect license plates in vehicles bboxes
try:
    ocr_model = ONNXPlateRecognizer(
        "global-plates-mobile-vit-v2-model", device="cuda"
    )  # Recognize license plates
except Exception as e:
    print("Error loading OCR model: ", e)
    ocr_model = ONNXPlateRecognizer(
        "global-plates-mobile-vit-v2-model", device="cpu"
    )  # Recognize license plates
# Results
unique_license_plates = {}


def annotate_frame(frame, results, texts):
    if results is None:
        return frame
    for box, text in zip(results.boxes, texts):
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
        cls = int(box.cls.cpu().numpy())
        label = (
            f"{text}"
            if text
            else f"{license_plate_model.names[cls]}"
        )
        draw_area(frame, (x1, y1, x2, y2))
        draw_text(frame, label, x1, y1)
    return frame


def preprocess(frame):
    return apply_clahe_to_frame(frame)


def predict(frame):
    return license_plate_model.track(frame, persist=True, verbose=False)[0]


def postprocess(frame, results, analysis_area):
    frame_area = frame.shape[0] * frame.shape[1]
    results = results[results.boxes.conf > CONF_TH] if results else None
    results = filter_detections_inside_area(
        results, analysis_area, frame_area, MIN_LICENSE_PLATE_AREA_PERCENTAGE
    )
    texts = extract_text_from_bounding_boxes(
        frame, results, ocr_model, unique_license_plates
    )
    frame = annotate_frame(frame, results, texts)
    draw_area(frame, analysis_area)
    return adaptive_resize(frame)


def process_frame(frame, analysis_area):
    frame = preprocess(frame)
    results = predict(frame)
    return postprocess(frame, results, analysis_area)


def main(input_type, input_path):
    min_occurences = MIN_OCCURENCES
    if input_type == 'video':
        video_info = sv.VideoInfo.from_video_path(video_path=input_path)
        cap = cv2.VideoCapture(input_path)

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
            target_path=input_path.replace(".mp4", "_labeled.mp4"),
            video_info=video_info,
        ) as s:
            while True:
                start = time()

                ret, frame = cap.read()
                if not ret:
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

    if input_type == 'image':
        min_occurences = 1
        frame = cv2.imread(input_path)
        h, w = frame.shape[:2]
        analysis_area = (
            ((CONTOUR_THICKNESS // 2) + 1),
            h // HEIGHT_PART_ANALYSYS,
            w - ((CONTOUR_THICKNESS // 2) + 1),
            (HEIGHT_PART_ANALYSYS - 1) * h // HEIGHT_PART_ANALYSYS,
        )
        frame = process_frame(frame, analysis_area)
        cv2.imshow("License Plate Recognition", frame)
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()

    save_recognized_plates(unique_license_plates, min_occurences)
