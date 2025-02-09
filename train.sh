#!/bin/sh

python cli.py fit \
    --data.train_data_path "training_images" \
    --data.val_data_path "training_images" \
    --data.batch_size 64 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "ctc" \
    --model.learning_rate 1e-3 \
    --trainer.max_epochs 30 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "Test_1"\
    --trainer.logger.project "OCR"