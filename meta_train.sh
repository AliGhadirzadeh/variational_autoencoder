#!/bin/bash

python3 train_eeg_data.py \
--eval \
--path-to-model ./ \
--path-to-data ./data/snippets.npy \
--model-filename vae_99999.mdl \
--num-epoch 1000 \
--snapshot 100 \
--beta-min 0 \
--beta-max 0.001
