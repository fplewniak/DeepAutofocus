#!/usr/bin/bash
##########################################
lr=0.00027760306206483
model=LaplacianNet
out=LaplacianNet_Kaiming_L2
image_size=512
epochs=24
batch=128
savefig=LaplacianNet_Kaiming_L2.png
title='Laplacian Kaiming L2 512 '
filelist=/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/CorrectFocusOnCells.csv
lambda2=0.0000148159971684519
init_weights=kaiming
blocks=6
weighted_loss=lorentz
nonlinear=LeakyReLU
nonlinearh=ReLU
channels=8
##########################################
.venv/bin/python train_model.py --lr $lr --blocks $blocks --model $model --out $out --image_size $image_size --epochs $epochs --batch_size $batch --savefig $savefig --title "$title" --filelist $filelist --lambda2 $lambda2 --init_weights $init_weights --weighted_loss $weighted_loss --nonlinear $nonlinear --nonlinearh $nonlinearh --channels $channels
