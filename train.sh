#!/bin/sh

python cli.py fit \
    --data.train_data_path "/hdd1t/mduc/data/train" \
    --data.val_data_path "/hdd1t/mduc/data/val" \
    --data.batch_size 32 \
    --data.num_workers 4 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-3 \
    --model.batch_max_length 50 \
    --model.save_dir "checkpoints/run50" \
    --trainer.max_epochs 10 \
    --trainer.val_check_interval 0.2 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "real-train-50"\
    --trainer.logger.project "OCR"\
    --trainer.log_every_n_steps 16