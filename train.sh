#!/bin/sh

python cli.py fit \
    --data.train_data_path "/hdd1t/mduc/data/train" \
    --data.val_data_path "/hdd1t/mduc/data/val" \
    --data.batch_size 16 \
    --data.num_workers 8 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-3 \
    --model.batch_max_length 50 \
    --validation_interval 4.0 \ #TODO: Fix argument parsing in lightning cli
    --model.save_dir "checkpoints/run50" \
    --trainer.max_epochs 10 \
    --trainer.check_val_every_n_epoch 1 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "train-50"\
    --trainer.logger.project "OCR"\
    --trainer.log_every_n_steps 16