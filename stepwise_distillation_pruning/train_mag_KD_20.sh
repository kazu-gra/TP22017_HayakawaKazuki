#20*1
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/train-pre/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_1 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*2
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_1/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_2 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*3
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_2/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_3 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*4
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_3/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_4 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*5
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_4/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_5 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*6
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_5/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_6 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*7
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_6/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_7 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*8
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_7/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_8 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*9
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_8/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_9 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*10
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_9/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_10 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*11
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_10/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_11 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*12
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_11/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_12 --teacher /workspace/Outputs/train-pre/weights/best.pt

#20*13
python train.py --epochs 100 --prune_mode Mag --sparsity 0.2 --pretrained  /workspace/Outputs/mag_KD_20_12/train/weights/best.pt --save_dir /workspace/Outputs/mag_KD_20_13 --teacher /workspace/Outputs/train-pre/weights/best.pt