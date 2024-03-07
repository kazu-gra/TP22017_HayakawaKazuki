#40*2
#python train.py --epochs 100 --prune_mode Mag --sparsity 0.4 --pretrained  /workspace/40/ours1_1best.pt --save_dir /workspace/Outputs/mag_ours_40_2_1 --teacher /workspace/40/ours1_1best.pt

#python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_2_1/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_2_2 --teacher /workspace/Outputs/train-pre/weights/best.pt

#40*3
python train.py --epochs 100 --prune_mode Mag --sparsity 0.4 --pretrained /workspace/Outputs/mag_ours_40_2_2/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_3_1 --teacher /workspace/Outputs/mag_ours_40_2_2/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_3_1/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_3_2 --teacher /workspace/40/ours1_1best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_3_2/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_3_3 --teacher /workspace/Outputs/train-pre/weights/best.pt

#40*4
python train.py --epochs 100 --prune_mode Mag --sparsity 0.4 --pretrained /workspace/Outputs/mag_ours_40_3_3/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_4_1 --teacher /workspace/Outputs/mag_ours_40_3_3/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_4_1/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_4_2 --teacher /workspace/Outputs/mag_ours_40_2_2/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_4_2/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_4_3 --teacher /workspace/40/ours1_1best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_4_3/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_4_4 --teacher /workspace/Outputs/train-pre/weights/best.pt

#40*5
python train.py --epochs 100 --prune_mode Mag --sparsity 0.4 --pretrained /workspace/Outputs/mag_ours_40_4_4/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_5_1 --teacher /workspace/Outputs/mag_ours_40_4_4/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_5_1/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_5_2 --teacher /workspace/Outputs/mag_ours_40_3_3/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_5_2/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_5_3 --teacher /workspace/Outputs/mag_ours_40_2_2/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_5_3/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_5_4 --teacher /workspace/40/ours1_1best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_5_4/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_5_5 --teacher /workspace/Outputs/train-pre/weights/best.pt

#40*6
python train.py --epochs 100 --prune_mode Mag --sparsity 0.4 --pretrained /workspace/Outputs/mag_ours_40_5_5/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_6_1 --teacher /workspace/Outputs/mag_ours_40_5_5/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_6_1/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_6_2 --teacher /workspace/Outputs/mag_ours_40_4_4/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_6_2/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_6_3 --teacher /workspace/Outputs/mag_ours_40_3_3/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_6_3/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_6_4 --teacher /workspace/Outputs/mag_ours_40_2_2/train/weights/best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_6_4/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_6_5 --teacher /workspace/40/ours1_1best.pt

python train.py --epochs 100 --pretrained /workspace/Outputs/mag_ours_40_6_5/train/weights/best.pt --save_dir /workspace/Outputs/mag_ours_40_6_6 --teacher /workspace/Outputs/train-pre/weights/best.pt
#python train.py --epochs 1 --prune_mode Mag --sparsity 0.6 --pretrained  /workspace/Outputs/train-pre/weights/best.pt --teacher /workspace/Outputs/train-pre/weights/best.pt --save_sir /workspace/Outputs/mag_60_1

#python train.py --pretrained  /workspace/Outputs/train-pre/weights/best.pt --teacher /workspace/Outputs/train-pre/weights/best.pt
