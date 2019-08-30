#!/bin/bash

python3 train_eeg_data2.py \
--train \
--path-to-model /home/sebgho/eeg_project/model/eeg_vae \
--path-to-data /home/sebgho/eeg_project/raw_data/data/snippets/snippets.npy \
--num-epoch 1000 \
--crossvalidation True \
--patience 3 \
--snapshot 100 \
--beta-min 0 \
--beta-max 0.001