#!/bin/sh

python test.py \
    --checkpoint "/home/qsvm/lib_ocr/checkpoints/runreal200/model_val_epoch_9_loss_0.7742_cer_0.1290_wer_0.5776.ckpt" \
    --image "/home/qsvm/dataset/eval/paper/IC03_867_png" \
    --batch_max_length 200 \
    --output "results.txt" 