import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
currenttime = 0
previoustime = 0

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

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
