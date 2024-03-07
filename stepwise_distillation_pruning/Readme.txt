・環境構築

イメージ取得
docker pull ultralytics/ultralytics:latest

コンテナ作成
docker run -it --ipc=host --gpus all -v 192.168.OOO.OO@ディレクトリ:/workspace --name コンテナ名 ultralytics/ultralytics:latest(image名)


・学習
基本的な学習コード
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/train-pre/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_1 --teacher /workspace/Outputs/train-pre/weights/best.pt
    --epochs：エポック数
        FTモデル：100（YOLOv8公式）
        Scratchモデル：500（同上）
    --prune_mode：枝刈り手法（Mag，SNIP）
    --sparsity：枝刈り率
    --pretrained：学習するモデル
    --save_dir：保存先のフォルダ
    --teacher：教師モデルに使用するファイル（.pt）



・評価
python val.py --pretrained /workspace/yolov8n.pt
    --pretrained：評価するモデル

↓評価結果例

(base) root@2021d63de7d7:/workspace# python val.py --pretrained /workspace/Outputs/train-pre/weights/best.pt
load
/workspace/Outputs/train-pre/weights/best.pt
Ultralytics YOLOv8.0.203 🚀 Python-3.10.13 torch-2.1.0 CUDA:2 (NVIDIA RTX A6000, 48677MiB)
                                                       CUDA:3 (NVIDIA RTX A6000, 48677MiB)
val: Scanning /usr/src/datasets/VOC/labels/test2007.cache... 4952 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4952/4952 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 39/39 [01:29<00:00,  2.31s/it]
                   all       4952      12032      0.816       0.76      0.828      0.625
             aeroplane       4952        285      0.922      0.849      0.918       0.68
               bicycle       4952        337      0.912      0.832      0.906      0.696
                  bird       4952        459      0.802      0.725      0.793      0.551
                  boat       4952        263      0.744      0.673      0.726      0.482
                bottle       4952        469      0.799       0.65      0.727      0.503
                   bus       4952        213       0.87      0.812      0.893      0.769
                   car       4952       1201      0.869      0.855      0.921      0.739
                   cat       4952        358      0.875      0.818      0.891      0.719
                 chair       4952        756      0.665      0.601       0.66       0.46
                   cow       4952        244      0.767      0.783      0.869      0.664
           diningtable       4952        206      0.756      0.752      0.791      0.623
                   dog       4952        489      0.864      0.793      0.872      0.675
                 horse       4952        348      0.921      0.853      0.921      0.731
             motorbike       4952        325      0.894      0.803      0.891      0.649
                person       4952       4528      0.889      0.793      0.898      0.641
           pottedplant       4952        480      0.694      0.475      0.571      0.331
                 sheep       4952        242      0.715       0.79      0.818      0.632
                  sofa       4952        239      0.685      0.757       0.79      0.655
                 train       4952        282      0.899      0.848      0.904      0.675
             tvmonitor       4952        308      0.786      0.728      0.805      0.624
Speed: 0.9ms preprocess, 1.6ms inference, 0.0ms loss, 3.3ms postprocess per image
Saving /usr/src/ultralytics/runs/detect/val39/predictions.json...

Evaluating pycocotools mAP using /usr/src/ultralytics/runs/detect/val39/predictions.json and /usr/src/datasets/VOC/annotations/instances_val2017.json...
pycocotools unable to run: /usr/src/datasets/VOC/annotations/instances_val2017.json file not found
Results saved to /usr/src/ultralytics/runs/detect/val39
0.8282452783984983



・可視化

python detect.py --pretrained /workspace/yolov8n.pt
    --pretrained：可視化するモデル
