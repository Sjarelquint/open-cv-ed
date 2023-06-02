import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from vidgear.gears import CamGear

stream = CamGear(source=0,
                 logging=True).start()  # YouTube Video URL as input
count = 0
while True:
    frame = stream.read()
    bbox, label, conf = cv.detect_common_objects(frame,model='yolov4-tiny')
    frame = draw_bbox(frame, bbox, label, conf)
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
stream.stop()
cv2.destroyAllWindows()
