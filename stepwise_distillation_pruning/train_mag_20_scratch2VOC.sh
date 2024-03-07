#python train.py --epochs 500 --save_dir /workspace/Outputs/Scratch_VOC/train_pre

#60*1
python train.py --epochs 500 --prune_mode Mag --sparsity 0.6  --pretrained /workspace/Outputs/Scratch_VOC/train_pre/train/weights/best.pt --save_dir /workspace/Outputs/Scratch_VOC/mag_60_1 

#60*2
python train.py --epochs 500 --prune_mode Mag --sparsity 0.6 --pretrained  /workspace/Outputs/Scratch_VOC/mag_60_1/train/weights/best.pt --save_dir /workspace/Outputs/Scratch_VOC/mag_60_2 

#60*3
python train.py --epochs 500 --prune_mode Mag --sparsity 0.6 --pretrained  /workspace/Outputs/Scratch_VOC/mag_60_2/train/weights/best.pt --save_dir /workspace/Outputs/Scratch_VOC/mag_60_3

