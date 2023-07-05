import cv2
import numpy as np
import hand_tracking as ht

dim = (600, 400)
fingers = cv2.imread("fingers2.jpg", 1)
fingers = cv2.resize(fingers, dsize=dim, interpolation=cv2.INTER_AREA)
rows = np.vsplit(fingers, 2)
cells = []
for row in rows:
    rowcell = np.hsplit(row, 5)
    for cell in rowcell:
        cells.append(cell)
# print(len(cells))
cap = cv2.VideoCapture(0)
detector = ht.handDetector()
tipIds = [4, 8, 12, 16, 20]
while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    #print(lmList)
    if len(lmList) != 0:
        fingers = []

        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        h, w, c = cells[0].shape
        frame[0:h, 0:w] = cells[totalFingers-1]
        cv2.rectangle(frame, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
    cv2.imshow('cell', frame)
    cv2.waitKey(1)
