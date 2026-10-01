#!/usr/bin/bash
##########################################
#lr=0.00027760306206483
lr=1e-03
model=LaplacianNet
out=LaplacianNet_Kaiming_L2
image_size=512
epochs=336
batch=24
savefig=LaplacianNet_Kaiming_L2.png
title='Laplacian 6 blocks, 4 channels, warm-up cosine annealing 512 '
filelist=/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/CorrectFocusOnCells.csv
lambda1=3e-05
#lambda1=0.0
#lambda2=0.0000148159971684519
lambda2=0.0
init_weights=kaiming
blocks=6
weighted_loss=lorentz
nonlinear=LeakyReLU
nonlinearh=ReLU
channels=4
dropout=0.4655
period=30
##########################################
.venv/bin/python train_model.py --lr $lr --blocks $blocks --model $model --out $out --image_size $image_size --epochs $epochs --batch_size $batch --savefig $savefig --title "$title" --filelist $filelist --lambda2 $lambda2 --init_weights $init_weights --weighted_loss $weighted_loss --nonlinear $nonlinear --nonlinearh $nonlinearh --channels $channels --dropout $dropout --warmup --period $period
