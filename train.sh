#!/bin/sh

python cli.py fit \
    --data.train_data_path "training_images" \
    --data.val_data_path "training_images" \
    --data.batch_size 32 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-3 \
    --trainer.max_epochs 30 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "attn-with-batch-norm-with-scale"\
    --trainer.logger.project "OCR"\
    --trainer.log_every_n_steps 8