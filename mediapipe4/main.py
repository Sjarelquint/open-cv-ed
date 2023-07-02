import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False,
                               max_num_faces=2,
                               refine_landmarks=False,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
currenttime = 0
previoustime = 0

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

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
