#!/bin/sh

python cli.py fit \
    --data.train_data_path "training_images" \
    --data.val_data_path "training_images" \
    --data.batch_size 32 \
    --model.backbone_name "vgg" \
    --model.seq_name None \
    --model.pred_name "ctc" \
    --model.learning_rate 1e-3 \
    --trainer.max_epochs 30 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "VGG-real-test"\
    --trainer.logger.project "OCR"