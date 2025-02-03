from src.inference.base_inference import BaseInference
from src.utils.utils import (
    adaptive_resize,
    extract_text_from_bounding_boxes,
    filter_detections_inside_area,
    draw_area,
)
from src.configs import (
    MIN_LICENSE_PLATE_AREA_PERCENTAGE,
    CONF_TH,
)


class InferenceApproach2(BaseInference):
    def process_frame(self, frame, analysis_area, min_plate_area):
        frame = self.preprocess(frame)
        results = self.predict(frame)
        return self.postprocess(frame, results, analysis_area, min_plate_area)

    def postprocess(
        self,
        frame,
        results,
        analysis_area,
        min_plate_area=MIN_LICENSE_PLATE_AREA_PERCENTAGE,
    ):
        frame_area = frame.shape[0] * frame.shape[1]
        results = results[results.boxes.conf > CONF_TH] if results else None
        results = filter_detections_inside_area(
            results, analysis_area, frame_area, min_plate_area
        )
        texts = extract_text_from_bounding_boxes(
            frame, results, self.ocr_model, self.unique_license_plates
        )
        frame = self.annotate_frame(frame, results, texts)
        draw_area(frame, analysis_area)
        return adaptive_resize(frame)


def main(input_type, input_path):
    min_plate_area = 0.00125
    inference = InferenceApproach2()
    inference.main(input_type, input_path, min_plate_area)
