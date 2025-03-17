#!/bin/sh

python cli.py fit \
    --data.train_data_path "/hdd1t/mduc/data/train" \
    --data.val_data_path "/hdd1t/mduc/data/val" \
    --data.batch_size 16 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-3 \
    --model.batch_max_length 200 \
    --trainer.max_epochs 30 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "test-model"\
    --trainer.logger.project "OCR"\
    --trainer.log_every_n_steps 8