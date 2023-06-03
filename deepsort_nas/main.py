import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
import numpy as np
import math
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.draw import draw_boxes

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

model = models.get("yolo_nas_s", pretrained_weights="coco")
cap = cv2.VideoCapture(0)
cfg_deep = get_config()
cfg_deep.merge_from_file('deep_sort_pytorch\configs\deep_sort.yaml')
deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                    min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE,
                    n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=False)
while True:
    xywh_bbox = []
    confs = []
    labs = []
    outputs = []
    ret, frame = cap.read()
    if not ret:
        break
    results = list(model.predict(frame))[0]
    bbox_xyxys = results.prediction.bboxes_xyxy.tolist()
    conf = results.prediction.confidence
    labels = results.prediction.labels.tolist()
    for bbox_xyxy, conf, cls in zip(bbox_xyxys, conf, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        classname = int(cls)
        class_name = classNames[classname]
        conf = math.ceil((conf * 100)) / 100

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        # cv2.putText(frame, str(class_name), (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
        cx, cy = int((x1 + x2)) / 2, int((y1 + y2) / 2)
        bbox_width = abs(x1 - x2)
        bbox_height = abs(y1 - y2)
        cxcywh = [cx, cy, bbox_width, bbox_height]
        xywh_bbox.append(cxcywh)
        confs.append(conf)
        labs.append(int(cls))
    xywhs = torch.tensor(xywh_bbox)
    confss = torch.tensor(confs)
    outputs = deepsort.update(xywhs, confss, labs, frame)
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -2]
        object_id = outputs[:, -1]
        draw_boxes(frame, bbox_xyxy, identities, object_id)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
