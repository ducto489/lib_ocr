#!/bin/sh

# python test.py \
#     --checkpoint "/home/qsvm/temp/lib_ocr/checkpoints/run200/model_val_epoch_7_loss_0.7515_cer_0.0525_wer_0.1175.ckpt" \
#     --data_root "/home/qsvm/dataset/eval/paper" \
#     --batch_size 2 \
#     --output_dir "results_lowercase"

python test.py \
    --checkpoint "/home/qsvm/temp/lib_ocr/checkpoints/run200/model_train_epoch_6.ckpt" \
    --data_root "/home/qsvm/dataset/eval/paper" \
    --batch_size 2 \
    --output_dir "results_epoch_6"

python test.py \
    --checkpoint "/home/qsvm/temp/lib_ocr/checkpoints/run200/model_val_epoch_6_loss_0.7923_cer_0.0547_wer_0.1233.ckpt" \
    --data_root "/home/qsvm/dataset/eval/paper" \
    --batch_size 2 \
    --output_dir "results_epoch_mid_6"

python test.py \
    --checkpoint "/home/qsvm/temp/lib_ocr/checkpoints/run200/model_train_epoch_4.ckpt" \
    --data_root "/home/qsvm/dataset/eval/paper" \
    --batch_size 2 \
    --output_dir "results_epoch_4"

python test.py \
    --checkpoint "/home/qsvm/temp/lib_ocr/checkpoints/run200/model_val_epoch_3_loss_0.9336_cer_0.0679_wer_0.1515.ckpt" \
    --data_root "/home/qsvm/dataset/eval/paper" \
    --batch_size 2 \
    --output_dir "results_epoch_mid_3"

python test.py \
    --checkpoint "/home/qsvm/temp/lib_ocr/checkpoints/run200/model_train_epoch_5.ckpt" \
    --data_root "/home/qsvm/dataset/eval/paper" \
    --batch_size 2 \
    --output_dir "results_epoch_5"

python test.py \
    --checkpoint "/home/qsvm/temp/lib_ocr/checkpoints/run200/model_val_epoch_5_loss_0.8127_cer_0.0589_wer_0.1291.ckpt" \
    --data_root "/home/qsvm/dataset/eval/paper" \
    --batch_size 2 \
    --output_dir "results_epoch_mid_5"