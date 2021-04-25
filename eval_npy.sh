#!/bin/bash

python eval_npy.py \
        --batch_size 256 \
        --model_folder 'models'  \
        --classifier Cnn1_3k_10 \
        --model_name Cnn1_3k_10_1e4_256_30 \
        --test_path "Data_HDF5/STEAD-STEAD+GEO_0.1_test.npy"