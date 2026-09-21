#!/usr/bin/bash
##########################################
lr=1e-4
model=LaplacianNet
out=LaplacianNet_Kaiming_L2
image_size=512
epochs=16
batch=128
savefig=LaplacianNet_Kaiming_L2.png
title='Laplacian Kaiming L2 resize 512'
filelist=/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/CorrectFocusOnCells.csv
lambda2=1e-4
init_weights=kaiming
##########################################
.venv/bin/python train_model.py --lr $lr --model $model --out $out --image_size $image_size --epochs $epochs --batch_size $batch --savefig $savefig --title $title --filelist $file_list --lambda2 $lambda2 --init_weights $init_weights
