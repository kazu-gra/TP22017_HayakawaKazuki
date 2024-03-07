from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='/workspace/Outputs/mag_60_1')
parser.add_argument('--dataset', type=str, default='VOC.yaml')
parser.add_argument('--sparsity', type=float, default=False)
parser.add_argument('--pretrained', type=str, default=False)
parser.add_argument('--teacher', type=str, default=False)
parser.add_argument('--init', type=str, default='/workspace/Outputs/init_pre.pt')
parser.add_argument('--prune_mode', type=str, default=False)
parser.add_argument('--score_flag', type=str, default=False) #SNIPを使用するときの勾配計算時のみTrueにする
parser.add_argument('--score_path', type=str, default=False) #SNIPを使用するときのスコアを保存するフォルダ



args = parser.parse_args()

# Load a model
model = YOLO(args.pretrained if args.pretrained else 'yolov8n.yaml')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data = args.dataset,
                      epochs = args.epochs,
                      device = [2, 3], #GPU番号
                      batch = 128,
                      init = args.init,
                      prune_mode = args.prune_mode,
                      project = args.save_dir,
                      sparsity = args.sparsity,
                      teacher = args.teacher,
                      score_flag = args.score_flag,
                      score_path = args.score_path
                      )