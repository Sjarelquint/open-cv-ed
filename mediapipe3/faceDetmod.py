import cv2
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence, self.model_selection)

    def find_face(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        if self.results.detections:
            if draw:
                for id, detection in enumerate(self.results.detections):
                    self.mpDraw.draw_detection(img, detection)
        return img
        #  # print(id, detection)
        #  # print(detection.score)
        #  # print(detection.location_data.relative_bounding_box)

    def find_position(self, img):
        bboxList = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                score = detection.score
                bboxList.append([id, bbox, score])
        # cv2.rectangle(img, bbox, (255, 0, 255), 2)
        # print(bbox)
        # cv2.putText(img, f'{int(detection.score[0] * 100)}%',
        #             (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
        #             2, (255, 0, 255), 2)
        return bboxList


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = faceDetector()
    while True:
        success, img = cap.read()
        img = detector.find_face(img)
        list = detector.find_position(img)
        if len(list) != 0:
            print(list)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
