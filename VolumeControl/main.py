import cv2
import time
import hand_tracking as ht
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

cap = cv2.VideoCapture(0)
WCam = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HCam = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# cap.set(3, 1080)
# cap.set(4, 720)
detector = ht.handDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol=0
volBar=400
volPer=0

pTime = 0
while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (x1, y1), 15, (0, 255, 255), -1)
        cv2.circle(frame, (x2, y2), 15, (0, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
        length = math.hypot((x2 - x1), (y2 - y1))
        vol = np.interp(length, [50, 400], [minVol, maxVol])
        volBar = np.interp(length, [50, 400], [400, 150])
        volPer=np.interp(length, [50, 400], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        # print(length)
        if length < 50:
            cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), -1)
    cv2.putText(frame, f'{int(volPer)}%', (100, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
