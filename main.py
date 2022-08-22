import cv2
import numpy as np
import torch
from torchvision import transforms
from norfair import Tracker, Video, Paths, draw_tracked_boxes

import result_processing as rp
from models.experimental import attempt_load
from utils.plots import output_to_target
from utils.datasets import letterbox
from utils.general import non_max_suppression

device = "cuda" if torch.cuda.is_available() else "cpu"

tracker = Tracker(
  distance_function= rp.iou,
  distance_threshold= 3.33,
)

paths_drawer = Paths(rp.center, color=(255,0,0), thickness=-1, radius=2, attenuation=0.1)

video = Video(input_path="tokyo.mkv")
# For camera
# video = Video(camera=0)

model = attempt_load("yolov7.pt")
_ = model.eval()

for frame in video:
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # image = letterbox(image, (frame.shape[1]), stride=64, auto=True)[0]
  image = letterbox(image, (1280), auto=True)[0]
  image_ = image.copy()
  image = transforms.ToTensor()(image)
  image = torch.tensor(np.array([image.numpy()]))
  image = image.to(device)
  image = image.float()

  with torch.no_grad():
    output, _ = model(image)
  
  output = non_max_suppression(output, iou_thres=0.3)
  output = output_to_target(output)
  im0 = image[0].permute(1, 2, 0) * 255
  im0 = im0.cpu().numpy().astype(np.uint8)
  
  im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
  detections = []
  
  for idx in range(output.shape[0]):
    xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
    xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

    detections.append([xmin, ymin, xmax, ymax, output[idx, 6]])
  
  output = rp.yolo_detections()
  output.xyxy = detections
  detections = rp.yolo_detections_to_norfair_detections(output)

  tracked_objects = tracker.update(detections=detections)

  draw_tracked_boxes(im0, tracked_objects, border_colors=[200])
  
  im0 = paths_drawer.draw(im0, tracked_objects)
  cv2.imshow("frame", im0)
  cv2.waitKey(1)
  video.write(im0)