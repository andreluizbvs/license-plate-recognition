import numpy as np
from ultralytics import YOLO

from src.inference.base_inference import BaseInference
from src.configs import (
    VEHICLE_WEIGHTS_PATH,
    VEHICLE_CLASS_IDS,
    MIN_CAR_AREA_PERCENTAGE,
    MIN_LICENSE_PLATE_AREA_PERCENTAGE,
    CONF_TH,
)
from src.utils.utils import (
    extract_text_from_bounding_boxes,
    filter_detections_inside_area,
    adaptive_resize,
    draw_area,
    draw_text,
)


class InferenceApproach1(BaseInference):
    def __init__(self):
        super().__init__(approach_type="1")
        self.vehicle_model = YOLO(VEHICLE_WEIGHTS_PATH)

    def predict(self, frame, analysis_area):
        vehicle_results = self.vehicle_model.track(frame, verbose=False)[0]
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
            license_plate_results = self.license_plate_model(
                vehicle_crop, verbose=False
            )[0]
            texts = self.postprocess_license_plate(
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

    def postprocess_license_plate(self, frame, results, analysis_area):
        frame_area = frame.shape[0] * frame.shape[1]
        results = results[results.boxes.conf > CONF_TH] if results else None
        results = filter_detections_inside_area(
            results, analysis_area, frame_area, MIN_LICENSE_PLATE_AREA_PERCENTAGE
        )
        return extract_text_from_bounding_boxes(
            frame, results, self.ocr_model, self.unique_license_plates
        )

    def postprocess(self, frame, license_plate_crops, analysis_area):
        for license_plate_crop in license_plate_crops:
            x1, y1, x2, y2, texts = license_plate_crop
            draw_area(frame, [x1, y1, x2, y2])
            if texts:
                draw_text(frame, f"{texts[0]}", x1, y1)
        draw_area(frame, analysis_area)
        return adaptive_resize(frame)

    def process_frame(self, frame, analysis_area, min_plate_area=MIN_LICENSE_PLATE_AREA_PERCENTAGE):
        frame = self.preprocess(frame)
        license_plate_crops = self.predict(frame, analysis_area)
        return self.postprocess(frame, license_plate_crops, analysis_area)


def main(input_type, input_path):
    inference = InferenceApproach1()
    inference.main(input_type, input_path)
