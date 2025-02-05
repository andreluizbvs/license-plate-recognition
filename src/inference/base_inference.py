from abc import ABC, abstractmethod
from time import time

import cv2
from fast_plate_ocr import ONNXPlateRecognizer
from ultralytics import YOLO

from src.utils.utils import (
    apply_clahe_to_frame,
    draw_area,
    draw_text,
    print_results,
    save_recognized_plates,
    show_approach_type,
    adaptive_resize
)
from src.configs import (
    CONTOUR_THICKNESS,
    SAVE_VIDEO,
    LICENSE_PLATE_WEIGHTS_PATH,
    HEIGHT_PART_ANALYSYS,
    MIN_OCCURENCES,
    MIN_LICENSE_PLATE_AREA_PERCENTAGE,
)


class BaseInference(ABC):
    def __init__(self, approach_type):
        self.license_plate_model = YOLO(LICENSE_PLATE_WEIGHTS_PATH)
        try:
            self.ocr_model = ONNXPlateRecognizer(
                "global-plates-mobile-vit-v2-model", device="cuda"
            )
        except Exception:
            self.ocr_model = ONNXPlateRecognizer(
                "global-plates-mobile-vit-v2-model", device="cpu"
            )
        self.unique_license_plates = {}
        self.approach_type = approach_type

    def preprocess(self, frame):
        return apply_clahe_to_frame(frame)

    def predict(self, frame):
        return self.license_plate_model.track(
            frame, persist=True, verbose=False
        )[0]

    @abstractmethod
    def postprocess(self):
        pass

    def annotate_frame(self, frame, results, texts):
        if results is None:
            return frame
        for box, text in zip(results.boxes, texts):
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
            cls = int(box.cls.cpu().numpy())
            label = (
                f"{text}" if text else f"{self.license_plate_model.names[cls]}"
            )
            draw_area(frame, (x1, y1, x2, y2))
            draw_text(frame, label, x1, y1)
        return frame

    @abstractmethod
    def process_frame(self):
        pass

    def analysis_area_calc(self, width, height):
        return (
            ((CONTOUR_THICKNESS // 2) + 1),
            height // HEIGHT_PART_ANALYSYS,
            width - ((CONTOUR_THICKNESS // 2) + 1),
            (HEIGHT_PART_ANALYSYS - 1) * height // HEIGHT_PART_ANALYSYS,
        )

    def main(
        self,
        input_type,
        input_path,
        min_plate_area=MIN_LICENSE_PLATE_AREA_PERCENTAGE,
    ):
        print("\n\nStarting License Plate Recognition...\n")
        min_occurences = MIN_OCCURENCES
        if input_type == "video":
            cap = cv2.VideoCapture(input_path)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            analysis_area = self.analysis_area_calc(w, h)

            print(f"Original video res. (width x height): {w}x{h}")
            print("Original video FPS: ", cap.get(cv2.CAP_PROP_FPS))
            print(f"Original video frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
            print(
                "Original video duration: ",
                cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                "seconds",
            )

            frame_counter = 0

            frame = cap.read()[1]
            first_frame = adaptive_resize(frame)
            output_path = input_path.split("/")[-1].split(".")[0] + "_out.avi"
            s = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                cap.get(cv2.CAP_PROP_FPS),
                (first_frame.shape[1], first_frame.shape[0]),
            )
            while True:
                start = time()

                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.process_frame(frame, analysis_area, min_plate_area)
                print_results(
                    self.unique_license_plates,
                    frame_counter,
                    min_occurences,
                )
                show_approach_type(self.approach_type, frame)
                cv2.imshow("License Plate Recognition", frame)

                if SAVE_VIDEO:
                    s.write(frame)

                frame_counter += 1
                total_proc_time = int((time() - start) * 1000)
                if cv2.waitKey(max(1, 33 - total_proc_time)) & 0xFF == ord("q"):
                    break
            print_results(
                self.unique_license_plates,
                frame_counter,
                min_occurences,
            )
            save_recognized_plates(self.unique_license_plates, min_occurences)
            cap.release()
            s.release()

        if input_type == "image":
            min_occurences = 1
            frame = cv2.imread(input_path)
            analysis_area = self.analysis_area_calc(
                frame.shape[1], frame.shape[0]
            )
            frame = self.process_frame(frame, analysis_area, min_plate_area)
            print_results(self.unique_license_plates, 0, min_occurences)
            save_recognized_plates(self.unique_license_plates, min_occurences)
            show_approach_type(self.approach_type, frame)
            cv2.imshow("License Plate Recognition", frame)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
