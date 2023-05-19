from ultralytics import YOLO
import numpy as np
import cv2
limits = [713, 436, 1112, 436]
totalCount = []
model = YOLO('yolov8n.pt')
results = model.track(source='Motorway.mp4', show=True, stream=True)
for r in results:
    boxes = np.array(r.boxes.xyxy.cpu(), dtype="int")
    ids = np.array(r.boxes.id.cpu(), dtype="int")
    classes = np.array(r.boxes.cls.cpu(), dtype="int")
    for idi, bbox, cls in zip(ids, boxes, classes):
        x1, y1, x2, y2 = bbox
        if limits[0] < x1 < limits[2] and limits[1] - 15 < y1 < limits[1] + 15:
            if totalCount.count(idi) == 0:
                totalCount.append(idi)
    print(len(totalCount))
