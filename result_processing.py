import numpy as np
from norfair.tracker import Detection

class yolo_detections:
  xyxy = []

def center(points):
  return [np.mean(np.array(points), axis=0)]

def iou(detection, tracked_object):
    box_a = np.concatenate([detection.points[0], detection.points[1]])
    box_b = np.concatenate(
        [tracked_object.estimate[0], tracked_object.estimate[1]])

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return 1 / iou if iou else (10000)

def yolo_detections_to_norfair_detections(yolo_detections):
  norfair_detections = []

  detections_as_xyxy = yolo_detections.xyxy
  
  for detection_as_xyxy in detections_as_xyxy:
    bbox = np.array(
      [
        [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
        [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
      ]
    )
    
    scores = np.array(
      [
        detection_as_xyxy[4].item(),
        detection_as_xyxy[4].item()
      ]
    )
    
    norfair_detections.append(
      Detection(points=bbox, scores=scores)
    )

  return norfair_detections
