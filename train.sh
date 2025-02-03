
python cli.py fit \
    --train_dir "training_images/images" \
    --val_dir "validation_images/images" \
    --backbone "resnet18" \
    --seq_module "bilstm" \
    --pred_module "ctc" \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --max_epochs 10
