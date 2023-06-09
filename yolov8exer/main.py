from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
import pandas as pd

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame)
        #  method 1
        # annotated_frame = results[0].plot()
        # cv2.imshow('frame', annotated_frame)

        # method 2
        # result = results[0]
        # bboxes = np.array(result.boxes.xyxy, dtype="int")
        # classes = np.array(result.boxes.cls, dtype="int")
        # for cls, bbox in zip(classes, bboxes):
        #     (x, y, x2, y2) = bbox
        #     cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        #     cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

        # method 3
        # for result in results:
        #     for r in result.boxes.data.tolist():
        #         x1, y1, x2, y2, score, class_id = r
        #         x1 = int(x1)
        #         x2 = int(x2)
        #         y1 = int(y1)
        #         y2 = int(y2)
        #         cls = int(class_id)
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        #         cv2.putText(frame, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

        # method 4
        # for r in results:
        #     boxes = r.boxes
        #     for box in boxes:
        #         x1, y1, x2, y2 = box.xyxy[0]
        #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #         w, h = x2 - x1, y2 - y1
        #         cls = int(box.cls[0])
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        #         cv2.putText(frame, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

        # method 5
        # for result in results:
        #     detections = sv.Detections.from_yolov8(result)
        #     labels = [
        #         f"{class_id} {confidence:0.2f}"
        #         for _, _, confidence, class_id,_
        #         in detections
        #     ]
        #
        #     frame = box_annotator.annotate(
        #         scene=frame,
        #         detections=detections,
        #         labels=labels
        #
        #     )

        # method 6
        # result = results[0].boxes.boxes
        # r = pd.DataFrame(result).astype("float")
        # for index, row in r.iterrows():
        #     x1 = int(row[0])
        #     y1 = int(row[1])
        #     x2 = int(row[2])
        #     y2 = int(row[3])
        #     cls = int(row[5])
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        #     cv2.putText(frame, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

        # method 7
        # result = list(results)[0]
        # bbox_xyxys = result.boxes.xyxy.tolist()
        # labels = result.boxes.cls.tolist()
        # for bbox_xyxy, cls in zip(bbox_xyxys, labels):
        #     bbox = np.array(bbox_xyxy)
        #     x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     classname = int(cls)
        #
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        #     cv2.putText(frame, str(classname), (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
