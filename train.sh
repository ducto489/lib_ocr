#!/bin/sh

python cli.py fit \
    --data.train_data_path "/home/qsvm/dataset/train" \
    --data.val_data_path "/home/qsvm/dataset/val" \
    --data.batch_size 64 \
    --data.num_workers 8 \
    --data.dali True \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-4 \
    --model.batch_max_length 50 \
    --model.save_dir "checkpoints/run50" \
    --trainer.max_epochs 2 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "train50-dali"\
    --trainer.logger.project "OCR"\
    --trainer.log_every_n_steps 16