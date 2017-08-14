#!/bin/bash

time python CNN/cnn_main.py --train_data_path=./cifar10/data_batch* \
                       --log_root=./tmp/CNN_base_model \
                       --train_dir=./tmp/CNN_base_model/train \
                       --model_name='CNN_base' \
                       --dataset='cifar10'

