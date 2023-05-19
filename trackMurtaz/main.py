import cv2
import numpy as np
import cvzone
import math
from ultralytics import YOLO
from sort import *

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
roi = np.array([[[677, 361], [985, 317], [1300, 442], [738, 544]]], np.int32)
limits = [713, 436, 1112, 436]
totalCount = []
cap = cv2.VideoCapture('Motorway.mp4')

model = YOLO("yolov8n.pt")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    ret, frame = cap.read()
    cv2.polylines(frame, [roi], True, (0, 0, 255), 3)
    if ret is False:
        break
    results = model(frame, conf=0.3, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            inside = cv2.pointPolygonTest(roi, (x1, y1), False)
            if inside == 1:
                if currentClass == 'car' or currentClass == 'bus' or currentClass == 'truck':
#                    cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
#                                       scale=2, thickness=1, offset=3)
#                    cvzone.cornerRect(frame, (x1, y1, w, h), l=5, rt=1)
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f' {int(Id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        if limits[0] < x1 < limits[2] and limits[1] - 15 < y1 < limits[1] + 15:
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    cv2.putText(frame, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
