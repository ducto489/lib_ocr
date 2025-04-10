# OCR - Optical Character Recognition

This repository contains a PyTorch Lightning implementation of an OCR (Optical Character Recognition) system with NVIDIA DALI data loading support.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
  - [Environment Setup](#environment-setup)
  - [Data Setup](#data-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)

## Features

- Multiple backbone architectures (ResNet, VGG, etc.)
- Sequence modeling options (BiLSTM)
- Multiple prediction methods (CTC, Attention)
- NVIDIA DALI data loading for high-performance training
- PyTorch Lightning for organized training workflow

## Setup

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/AnyGlow/lib_ocr.git
cd lib_ocr
```

2. Install dependencies:
```bash
pip install -r requirements.txt

### Data Setup

#### Option 1: Download from Kaggle

1. Install the Kaggle CLI:
```bash
pip install kaggle
```

2. Set up Kaggle credentials:
   - Go to your Kaggle account, then to 'Account' tab
   - Scroll down to API section and click 'Create New API Token'
   - Place the downloaded `kaggle.json` file in `~/.kaggle/`
   - Set proper permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Download the OCR dataset:
```bash
# Replace with your specific dataset command
kaggle datasets download -d [dataset-name]
unzip [dataset-name].zip -d /path/to/data/folder
```

4. Prepare the data in the required format:
```
/path/to/data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── tgt.csv
└── val/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── tgt.csv
```

The `tgt.csv` should contain image names and their corresponding text labels.

## Training

Start training with the following command:

```bash
python cli.py fit \
    --data.train_data_path "/path/to/your/train" \
    --data.val_data_path "/path/to/your/val" \
    --data.batch_size 16 \
    --data.num_workers 4 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-3 \
    --model.batch_max_length 50 \
    --model.save_dir "checkpoints/run50" \
    --trainer.max_epochs 10 \
    --trainer.val_check_interval 0.1 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "real-train-50" \
    --trainer.logger.project "OCR" \
    --trainer.log_every_n_steps 16
```

### Training Parameters

- `--data.train_data_path`: Path to training data directory
- `--data.val_data_path`: Path to validation data directory
- `--data.batch_size`: Batch size for training
- `--data.num_workers`: Number of data loading workers
- `--model.backbone_name`: CNN backbone architecture (options: resnet18, vgg)
- `--model.seq_name`: Sequence module (options: bilstm, none)
- `--model.pred_name`: Prediction module (options: ctc, attn)
- `--model.learning_rate`: Learning rate (default: 1e-3)
- `--model.batch_max_length`: Maximum length of text sequences
- `--model.save_dir`: Directory to save checkpoints
- `--trainer.max_epochs`: Maximum number of training epochs
- `--trainer.val_check_interval`: Validation check frequency (as float or int)
- `--trainer.logger`: Logger to use (e.g., WandbLogger)
- `--trainer.logger.name`: Name of the experiment in the logger
- `--trainer.logger.project`: Project name in the logger
- `--trainer.log_every_n_steps`: Logging frequency during training