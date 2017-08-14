#!/bin/bash
                    
python CNN/cnn_main.py --eval_data_path=./cifar10/test_batch.bin \
                       --log_root=./tmp/CNN_base_model \
                       --eval_dir=./tmp/CNN_base_model/test \
                       --dataset='cifar10' \
                       --mode=eval \
                      --model_name='CNN_base'
