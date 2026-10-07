#!/usr/bin/bash
##########################################
#lr=0.00027760306206483
lr=1e-03
model=LaplacianNet
out=LaplacianNet
image_size=512
epochs=32
batch=64
savefig=LaplacianNet
title='Laplacian 7 blocks, 6 channels, warm-up cosine annealing 512 '
filelist=/data3/DeepAutoFocus/20250610_Nikon_zStacks_W52_YAK1-09/CorrectFocusOnCells.csv
lambda1=3e-05
#lambda2=0.0000148159971684519
lambda2=0.0
init_weights=kaiming
blocks=7
weighted_loss=lorentz
nonlinear=LeakyReLU
nonlinearh=ReLU
channels=6
dropout=0.4655
period=10
warmup=10
logdir='runs'
##########################################
.venv/bin/python train_model.py --lr $lr --blocks $blocks --model $model --out $out --image_size $image_size --epochs $epochs --batch_size $batch --savefig $savefig --title "$title" --filelist $filelist --lambda1 $lambda1 --lambda2 $lambda2 --init_weights $init_weights --weighted_loss $weighted_loss --nonlinear $nonlinear --nonlinearh $nonlinearh --channels $channels --dropout $dropout --warmup $warmup --period $period
--logdir $logdir