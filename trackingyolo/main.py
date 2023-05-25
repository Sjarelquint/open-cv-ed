from ultralytics import YOLO
import numpy as np
import cv2
import supervision as sv

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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
LINE_START = sv.Point(0, 436)
LINE_END = sv.Point(1200, 436)


# limits = [713, 436, 1112, 436]
# totalCount = []
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

model = YOLO('yolov8m.pt')
results = model.track(source='Motorway.mp4', stream=True)
for result in results:
    frame = result.orig_img
    detections = sv.Detections.from_yolov8(result)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[(detections.class_id == 7)]

    labels = [
        f"{tracker_id} {classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]

    frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels

    )

    # for r in results:
    #     boxes = np.array(r.boxes.xyxy.cpu(), dtype="int")
    #     ids = np.array(r.boxes.id.cpu(), dtype="int")
    #     classes = np.array(r.boxes.cls.cpu(), dtype="int")
    #     for idi, bbox, cls in zip(ids, boxes, classes):
    #         x1, y1, x2, y2 = bbox
    #         if limits[0] < x1 < limits[2] and limits[1] - 15 < y1 < limits[1] + 15:
    #             if totalCount.count(idi) == 0:
    #                 totalCount.append(idi)
    #     print(len(totalCount))
    line_counter.trigger(detections=detections)
    line_annotator.annotate(frame=frame, line_counter=line_counter)
    cv2.imshow("yolov8", frame)

    if cv2.waitKey(30) == 27:
        break
