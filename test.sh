#!/bin/sh

python test.py \
    --checkpoint "/hdd1t/mduc/ocr/lib_ocr/training/lib_ocr/checkpoints/model_train_epoch_8.ckpt" \
    --image "/hdd1t/mduc/data/eval/paper/IC03_867_png" \
    --batch_max_length 200 \
    --output "results.txt" 