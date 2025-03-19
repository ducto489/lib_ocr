#!/bin/sh

python cli.py fit \
    --data.train_data_path "/hdd1t/mduc/data/train" \
    --data.val_data_path "/hdd1t/mduc/data/val" \
    --data.batch_size 16 \
    --data.num_workers 12 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-3 \
    --model.batch_max_length 200 \
    --data.batch_max_length 200 \
    --validation_interval 4.0 \
    --model.save_dir "checkpoints/run200" \
    --trainer.max_epochs 10 \
    --trainer.check_val_every_n_epoch 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "train-200"\
    --trainer.logger.project "OCR"\
    --trainer.log_every_n_steps 16