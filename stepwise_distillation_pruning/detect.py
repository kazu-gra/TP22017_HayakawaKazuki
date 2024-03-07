'''from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define path to directory containing images and videos for inference
source = '/usr/src/datasets/coco/images/test2017'

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects'''

from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='/workspace/yolov8n.pt')
args = parser.parse_args()

pretrained = args.pretrained

# Load a pretrained YOLOv8n model
model = YOLO(pretrained)

# Run inference on 'bus.jpg' with arguments
model.predict('/usr/src/datasets/VOC/images/test2007',save=True, imgsz=320, conf=0.5)
#model.predict('/usr/src/datasets/coco/images/test2017/000000000001.jpg', save=True, imgsz=320, conf=0.5)