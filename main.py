import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

# TODO: adjust zone polygon
ZONE_POLYGON = np.array([
    [0,0],
    [0.5, 0],
    [0.5, 1],
    [0,1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

    polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=polygon)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id, [1,2,3,5,7])]
        detections = detections[detections.confidence > 0.5]
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone_detects = zone.trigger(detections=detections)
        count_zone_detects = np.count_nonzero(zone_detects) # *Contains the number of detected objects inside zone

        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("yolov8", frame)


        if (cv2.waitKey(30) == 27):
            break


if __name__ == '__main__':
    main()