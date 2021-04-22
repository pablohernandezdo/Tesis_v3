#!/bin/bash

python train_npy.py \
        --lr 1e-4 \
        --epochs 1 \
        --batch_size 256 \
        --patience 30 \
        --model_folder 'models'  \
        --classifier Cnn1_3k_10 \
        --model_name Cnn1_3k_10_1e4_256_30_SG \
        --train_path "Data_HDF5/STEAD_STEAD_0.8_train.npy" \
        --val_path "Data_HDF5/STEAD_STEAD_0.1_val.npy"