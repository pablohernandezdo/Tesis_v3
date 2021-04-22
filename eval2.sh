#!/bin/bash

python eval_hdf5.py \
        --batch_size 256 \
        --model_folder 'models'  \
        --classifier Cnn1_3k_10 \
        --model_name Cnn1_3k_10_1e4_256_30_SS \
        --train_path "Data_HDF5/STEAD-STEAD_0.8_train.hdf5" \
        --test_path "Data_HDF5/STEAD-STEAD_0.1_test.hdf5"