#!/bin/bash

python train.py \
        --lr 1e-5 \
        --epochs 1 \
        --batch_size 256 \
        --patience 30 \
        --model_folder 'models'  \
        --classifier Cnn1_3k_10 \
        --model_name Cnn1_3k_10_1e4_256_30_SS \
        --train_path "Data_HDF5/STEAD-STEAD_0.8_train.hdf5" \
        --val_path "Data_HDF5/STEAD-STEAD_0.1_val.hdf5"