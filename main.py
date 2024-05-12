import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import serial
import serial.tools.list_ports
import sys
# ============================= GLOBAL VARIABLES =============================

car_thresh = 6
# TODO: adjust zone polygon
ZONE_POLYGON = np.array([
    [0,0],
    [0.5, 0],
    [0.5, 1],
    [0,1]
])

def port_identifier():
    ports = serial.tools.list_ports.comports()
    portsList = {}
    print("---------------------- ALL AVAILABLE COM PORTS ----------------------")
    print(" ")
    for port, desc, hwid in sorted(ports):
       
        print("{}: {} [{}]".format(port, desc, hwid))
        portsList[port] = desc
    return portsList

def getSerial(COM, coms):
    if COM in coms:
        ser = serial.Serial('COM12', 9600)
        Serial_Command = input("Command: ")
        if Serial_Command == "OFF":
            ser.close()
            return None
        else:
            ser.write(bytearray(Serial_Command, 'ascii'))
            return ser

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def main():
    isGo = False
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # ============================= VIDEO CAPTURE INITIALIZATION =============================

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    # ============================= YOLOV8 MODEL INITIALIZATION =============================

    model = YOLO("yolov8l.pt")

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

    polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=polygon)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

    # ============================= SERIAL PORT INITIALIZATION =============================

    COM_ports = port_identifier()

    COM = input("COM Port: ")
    
    ser = getSerial(COM, COM_ports)

    # ============================= MAIN PROGRAM EXECUTION =============================
    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id, [1,2,3,5,7])]
        # detections = detections[detections.confidence > 0.5]
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]
        frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone_detects = zone.trigger(detections=detections)
        count_zone_detects = np.count_nonzero(zone_detects)

        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("yolov8", frame)
        
        if (count_zone_detects > car_thresh and not isGo):
            isGo = True
            ser.write(bytearray('GO\n','ascii'))
            print('GO')
        elif (count_zone_detects <= car_thresh and isGo):
            isGo = False
            ser.write(bytearray('STOP\n','ascii'))
            print('STOP')

        if (cv2.waitKey(30) == 27):
            ser.close()
            print("closing port")
            break


if __name__ == '__main__':
    main()

# ============================= ARDUINO CODE =============================
'''
#define GREEN 2
#define YELLOW 3
#define RED 4

void setup() {
  Serial.begin(9600);
  pinMode(GREEN, OUTPUT);
  pinMode(YELLOW, OUTPUT);
  pinMode(RED, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readChar;
    if (input == "GO") {
      LightSequence(input);
      Serial.println(input);
    } else if (input == "STOP") {
      LightSequence(input);
      Serial.println(input);
    }
  }
}

void LightSequence(String status) {
  if (status == "GO") {
    digitalWrite(GREEN, HIGH);
    digitalWrite(YELLOW, LOW);
    digitalWrite(RED, LOW);
  } else if (status == "STOP") {
    digitalWrite(GREEN, LOW);
    digitalWrite(YELLOW, HIGH);
    digitalWrite(RED, LOW);
    UserDelay(1000);
    digitalWrite(GREEN, LOW);
    digitalWrite(YELLOW, LOW);
    digitalWrite(RED, HIGH);
  }
}

void UserDelay(int time) {
  for (int i = 0; i < time; i++) {
    delay(1);

    if (Serial.available()) {
      break;
    }
  }
}
'''