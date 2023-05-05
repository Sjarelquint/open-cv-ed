from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n-seg.pt')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("frame", annotated_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
