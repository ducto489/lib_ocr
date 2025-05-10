#!/bin/sh

python cli.py fit \
    --data.train_data_path "/home/qsvm/dataset/train" \
    --data.val_data_path "/home/qsvm/dataset/val" \
    --data.batch_size 64 \
    --data.num_workers 8 \
    --data.dali True \
    --data.frac 1 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-4 \
    --model.batch_max_length 200 \
    --model.save_dir "checkpoints/runreal200" \
    --trainer.max_epochs 10 \
    --trainer.val_check_interval 0.5 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "real-dali-200"\
    --trainer.logger.project "OCR"\
    --trainer.log_every_n_steps 16