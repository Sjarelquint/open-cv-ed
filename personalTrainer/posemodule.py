import cv2
import mediapipe as mp
import time
import math

class PoseEstimator():
    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def findPose(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return frame

    def findPosition(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
        return self.lmList

    def findAngle(self, frame, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        print(angle)
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.line(frame, (x3, y3), (x2, y2), (0, 255, 255), 3)
            cv2.circle(frame, (x1, y1), 15, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 15, (0, 255, 0), -1)
            cv2.circle(frame, (x3, y3), 15, (0, 255, 0), -1)
            cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def main():
    currenttime = 0
    previoustime = 0
    cap = cv2.VideoCapture(0)
    estimator = PoseEstimator()
    while True:
        ret, frame = cap.read()
        frame = estimator.findPose(frame)
        lmList = estimator.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[4])

        currenttime = time.time()
        fps = int(1 / (currenttime - previoustime))
        previoustime = currenttime
        cv2.putText(frame, f'FPS {fps}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
