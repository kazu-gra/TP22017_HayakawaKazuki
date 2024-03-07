from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('--pretrained', type=str, default='/usr/src/ultralytics/runs/detect/train15/weights/last.pt')
parser.add_argument('--pretrained', type=str, default='/workspace/Outputs/train-pre/weights/best.pt')
args = parser.parse_args()

pretrained = args.pretrained

# Load a model
model = YOLO(pretrained)  # load an official model
#model = YOLO('path/to/best.pt')  # load a custom model
print(pretrained)
# Validate the model
metrics = model.val(device = [2, 3],
                    batch = 128)
                    #project = '/workspace/Outputs')  # no arguments needed, dataset and settings remembered
print(metrics.box.map50) #mAP50を出力(小数点第２位まで)
