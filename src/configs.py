from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Constants
GREEN = (0, 255, 0)
CONTOUR_THICKNESS = 5
RESIZE_FACTOR = 2
SAVE_VIDEO = False
VEHICLE_WEIGHTS_PATH = str(BASE_DIR / 'weights' / 'yolo11n.pt')
LICENSE_PLATE_WEIGHTS_PATH = str(BASE_DIR / 'weights' / 'license_plate_detector.pt')
CONF_TH = 0.2  # Confidence threshold to show the annotation and count the object
HEIGHT_PART_ANALYSYS = 1000  # Analysis area height (the higher the number, the bigger the part of the frame to be analyzed. if its is "12", then the first and the last 1/12 of the frame will not be analyzed)
IOU_PROPORTION = 0.8  # Proportion of the object that has to be inside the analysis area to be considered
MIN_CAR_AREA_PERCENTAGE = 0.025  # Minimum car area percentage of the input image area
MIN_LICENSE_PLATE_AREA_PERCENTAGE = 0.05  # Minimum license plate area percentage of the input image area
MIN_OCCURENCES = 20  # Minimum number of occurences to be considered a correctly recognized license plate
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO vehicle class IDs (car, motorcycle, bus, truck)