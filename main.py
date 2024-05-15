import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
from util import *

# ============================= GLOBAL VARIABLES =============================
ZONE_POLYGON = np.array([
    [0.17,.5],
    [0.78, 0.2],
    [1, 0.43],
    [0.33,0.95]
])


def main():
  # ============================= CONFIGURATION =============================
  max_time = 0                        # max time for countdown GO and STOP
  start_count = False                 # traffic counter state
  frame_count = 0
  car_detection_total = 0
  car_detection_avg = 0               # Average number of cars detected
  isGo = False                        # Traffic light state
  num_frames = 5
  args = parse_arguments()
  frame_width, frame_height = args.webcam_resolution
  
  go_time = 15                        # Number of seconds for GREEN light
  stop_time = 60                      # Number of seconds for RED light
  
  prev_car_count = 0
  car_weight = 0.05                   # Decrement ratio per car
  
  
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
  if ser:
    ser.write(bytearray('STOP\n','ascii'))
  

  # ============================= MAIN PROGRAM EXECUTION =============================
  while True:
    # --------------------------- Get Predictions from Yolov8 ---------------------------
    ret, frame = cap.read()
    result = model(frame, agnostic_nms=True, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(result)
    detections = detections[np.isin(detections.class_id, [1,2,3,5,7])]      # filter detections to vehicles only
    
    
    # --------------------------- Initialize Annotations ---------------------------
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

    # Show frame with annotations
    cv2.imshow("yolov8", frame)

    # --------------------------- Traffic Light Logic ---------------------------
    if (frame_count >= num_frames):
      car_detection_avg = round(car_detection_total / frame_count)
      car_detection_total = 0
      frame_count = 0
    else:
      car_detection_total += count_zone_detects
      frame_count += 1
    
    
    if car_detection_avg > 0 and not start_count:
      max_time = time.time() + stop_time
      start_count = True
    elif car_detection_avg < 1 and start_count and not isGo:
      max_time = 0
      start_count = False
      
    print(max_time - time.time())
    
    if (max_time - time.time()) > 5 and not isGo and prev_car_count < car_detection_avg:
          max_time -= ((stop_time * car_weight) * car_detection_avg)
          prev_car_count = car_detection_avg
    
    if time.time() > max_time and start_count:
      if (not isGo):
          isGo = True
          if ser:
              ser.write(bytearray('GO\n','ascii'))
          print("GO")
          max_time = time.time() + go_time
          prev_car_count = 0
      else:
          isGo = False
          if ser: 
              ser.write(bytearray('STOP\n','ascii'))
          print("STOP")
          
          max_time = time.time() + stop_time
    
    
    # Close window if 'Esc' key is pressed
    if (cv2.waitKey(30) == 27):
      if ser:
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