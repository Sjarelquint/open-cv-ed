import cv2
import mediapipe as mp
import time


class fmmode():
    def __init__(self, static_image_mode=False, max_num_faces=2, refine_landmarks=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces,
                                                 self.refine_landmarks,
                                                 self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFace(self, frame):
        listpos = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(frameRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec,
                                           self.drawSpec)
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    listpos.append([id, x, y])

        return listpos, frame


def main():
    currenttime = 0
    previoustime = 0
    cap = cv2.VideoCapture(0)
    estimator = fmmode()
    while True:
        ret, frame = cap.read()
        listpos, frame = estimator.findFace(frame)
        if len(listpos) != 0:
            print(listpos)
        currenttime = time.time()
        fps = int(1 / (currenttime - previoustime))
        previoustime = currenttime
        cv2.putText(frame, f'FPS {fps}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
