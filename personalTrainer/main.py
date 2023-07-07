import cv2
import mediapipe as mp
import time
import posemodule as pm
import numpy as np

cap = cv2.VideoCapture(0)
estimator = pm.PoseEstimator()

currenttime = 0
previoustime = 0
count = 0
dir = True
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = estimator.findPose(frame)
    lmList = estimator.findPosition(frame, draw=False)
    if len(lmList) != 0:
        angle = estimator.findAngle(frame, 12, 14, 16)
        # per = np.interp(angle, (210, 310), (0, 100))
        # bar = np.interp(angle, (220, 310), (650, 100))
        # # print(angle, per)
        # # Check for the dumbbell curls
        # color = (255, 0, 255)
        # if per == 100:
        #     color = (0, 255, 0)
        #     if dir == 0:
        #         count += 0.5
        #         dir = 1
        # if per == 0:
        #     color = (0, 255, 0)
        #     if dir == 1:
        #         count += 0.5
        #         dir = 0
        # print(count)
        # # Draw Bar
        # cv2.rectangle(frame, (1100, 100), (1175, 650), color, 3)
        # cv2.rectangle(frame, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        # cv2.putText(frame, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
        #             color, 4)
        # # Draw Curl Count
        # cv2.rectangle(frame, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        # cv2.putText(frame, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
        #             (255, 0, 0), 25)
        if angle < 90 and dir == True:
            count += 1
            dir = False
        elif angle > 160:
            dir = True
        cv2.putText(frame, str(count), (45, 500), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255, 0, 0), 25)

    print(count)
    currenttime = time.time()
    fps = int(1 / (currenttime - previoustime))
    previoustime = currenttime
    cv2.putText(frame, f'FPS {fps}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()
