import re
from time import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from fast_plate_ocr import ONNXPlateRecognizer


# Constants
GREEN = (0, 255, 0)
CONTOUR_THICKNESS = 5
RESIZE_FACTOR = 2
SAVE_VIDEO = False
LICENSE_PLATE_WEIGHTS_PATH = "./weights/license_plate_detector.pt"
VIDEO_PATH = "./data/sample2.mp4"
CONF_TH = 0.2  # Confidence threshold to show the annotation and count the object
HEIGHT_PART_ANALYSYS = 1000 # Analysis area height (the higher the number, the bigger the part of the frame to be analyzed. if its is "12", then the first and the last 1/12 of the frame will not be analyzed)
IOU_PROPORTION = 0.8  # Proportion of the object that has to be inside the analysis area to be considered
MIN_PLATE_AREA_PERCENTAGE = 0.00125  # Minimum license plate area percentage of the input image area
MIN_OCCURENCES = 15  # Minimum number of occurences to be considered a correctly recognized license plate

# Global instances
license_plate_model = YOLO(LICENSE_PLATE_WEIGHTS_PATH) # Detect license plates in vehicles bboxes
ocr_model = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model', device='cpu') # Recognize license plates

# Results
unique_license_plates = {}



def clean_license_plate(text):
    """
    Removes characters that don't normally appear in license plates.
    Keeps only alphanumeric characters and hyphens.
    """
    # Define a regex pattern to match valid license plate characters
    pattern = re.compile(r'[^A-Za-z0-9-]')
    
    # Substitute invalid characters with an empty string
    cleaned_text = pattern.sub('', text)
    
    return cleaned_text


def adaptive_resize(frame, max_height=720, max_width=1280):
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

    # Calculate intersection area between object and analysis area
    inter_x1 = np.maximum(x1, x_min)
    inter_y1 = np.maximum(y1, y_min)
    inter_x2 = np.minimum(x2, x_max)
    inter_y2 = np.minimum(y2, y_max)
    inter_area = (
        np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    )

    return inter_area >= IOU_PROPORTION * object_area


def preprocess(frame):
    return apply_clahe_to_frame(frame)


def predict(frame):
    return license_plate_model.track(frame, persist=True, verbose=False)[0]


def filter_detections_inside_area(results, analysis_area, frame_area):
    inside_area_indices = is_inside_predefined_area(results.boxes.xyxy.cpu().numpy(), analysis_area)
    valid_indices = inside_area_indices
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
        plate_area = (x2 - x1) * (y2 - y1)
        if plate_area < MIN_PLATE_AREA_PERCENTAGE * frame_area:
            valid_indices[i] = False

    if np.any(valid_indices):
        return results[valid_indices]
    return None


def annotate_frame(frame, results, texts):   
    for box, text in zip(results.boxes, texts):
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
        conf = float(box.conf.cpu().numpy())
        cls = int(box.cls.cpu().numpy())
        label = f"{text} {conf:.2f}" if text else f"{license_plate_model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, CONTOUR_THICKNESS)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
    return frame


def draw_analysis_area(frame, analysis_area):
    cv2.rectangle(
        frame,
        (analysis_area[0], analysis_area[1]),
        (analysis_area[2], analysis_area[3]),
        GREEN,
        CONTOUR_THICKNESS,
    )


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

    cv2.imshow("License Plate Recognition", image)
    
    return image


def extract_text_from_bounding_boxes(frame, results):
    texts = []
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
                unique_license_plates[cleaned_text] = unique_license_plates.get(cleaned_text, 0) + 1
        else:
            texts.append("")
    return texts


def postprocess(frame, results, analysis_area):
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width

    results = results[results.boxes.conf > CONF_TH] if results else None
    
    if results is not None:
        results = filter_detections_inside_area(results, analysis_area, frame_area)
    if results is not None:
        texts = extract_text_from_bounding_boxes(frame, results)
        frame = annotate_frame(frame, results, texts)
    
    draw_analysis_area(frame, analysis_area)
    return adaptive_resize(frame)


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

    with sv.VideoSink(
        target_path=VIDEO_PATH.replace(".mp4", "_labeled.mp4"),
        video_info=video_info,
    ) as s:
        while True:
            start = time()

            ret, frame = cap.read()
            if not ret:
                break
            
            frame = preprocess(frame)
            results = predict(frame)
            frame = postprocess(frame, results, analysis_area)

            if SAVE_VIDEO:
                s.write_frame(frame=frame)

            total_proc_time = int((time() - start) * 1000)

            cv2.imshow("License Plate Recognition", frame)
            frequent_plates = {plate: count for plate, count in unique_license_plates.items() if count >= MIN_OCCURENCES}
            print("Recognized License Plates:", list(frequent_plates.keys()))
            if cv2.waitKey(max(1, 33 - total_proc_time)) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
