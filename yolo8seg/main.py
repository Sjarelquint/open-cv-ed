from yolo_segmentation import YOLOSegmentation
import cv2

ys = YOLOSegmentation("yolov8m-seg.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    bboxes, classes, segmentations, scores = ys.detect(frame)
    print(segmentations)
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        (x, y, x2, y2) = bbox
        # cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
        cv2.putText(frame, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
