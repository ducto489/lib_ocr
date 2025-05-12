# OCR - Optical Character Recognition Engine

This repository provides a flexible and high-performance Optical Character Recognition (OCR) system built with PyTorch Lightning and accelerated using NVIDIA DALI for data loading.

## Overview

This project implements an end-to-end OCR pipeline, featuring:

*   **Multiple Architectures:** Easily swap CNN backbones (ResNet, VGG) and sequence modeling layers (BiLSTM).
*   **Prediction Methods:** Supports both CTC (Connectionist Temporal Classification) and Attention-based decoding.
*   **High-Performance Data Loading:** Integrates NVIDIA DALI for significantly faster data preprocessing and loading, especially beneficial on NVIDIA GPUs.
*   **Structured Training:** Leverages PyTorch Lightning for organized, reproducible training workflows, including logging (e.g., Wandb) and checkpointing.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
  - [Environment](#environment)
  - [Get the Code](#get-the-code)
  - [Install Dependencies](#install-dependencies)
  - [Download Data](#download-data)
  - [Data Structure](#data-structure)
- [Data Details](#data-details)
  - [Character Normalization](#character-normalization)
  - [Datasets Used](#datasets-used)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

## Features

*   **Flexible Model Configuration:** Choose from various backbones (ResNet, VGG), sequence modules (BiLSTM), and prediction heads (CTC, Attention).
*   **NVIDIA DALI Integration:** Enables high-throughput data loading pipelines on GPUs, reducing CPU bottlenecks.
*   **PyTorch Lightning Workflow:** Organised code, multi-GPU support, logging, checkpointing, and more out-of-the-box.
*   **Standard Dataset Support:** Includes examples for training on common OCR datasets.
*   **Clear Inference & Evaluation:** Scripts provided for running predictions and benchmarking performance.

## Setup

Follow these steps to set up your environment and download the necessary data.

### Environment

It's recommended to use a virtual environment. This project uses Python 3.11.

```bash
conda create -n ocr python=3.11
conda activate ocr
```

### Get the Code

```bash
git clone https://github.com/AnyGlow/lib_ocr.git
cd lib_ocr
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Data

The primary dataset can be downloaded from Hugging Face. Choose one of the following methods:

**Method 1: Using `wget`**

```bash
# Download the dataset archive
wget https://huggingface.co/datasets/ducto489/ocr_datasets/resolve/main/output.zip

# Extract to your desired data location
unzip output.zip -d /path/to/your/data/directory
```

**Method 2: Using Hugging Face CLI**

```bash
# Install Hugging Face Hub if you don't have it
pip install huggingface_hub

# Download and extract the dataset
huggingface-cli download ducto489/ocr_datasets output.zip --repo-type dataset --local-dir .
unzip output.zip -d /path/to/your/data/directory
```

Replace `/path/to/your/data/directory` with the actual path where you want to store the data.

### Data Structure

Organize the downloaded data into the following structure:

```
/path/to/your/data/directory/
├── train/
│   ├── images/         # Directory containing training images
│   │   ├── image1.jpg
│   │   └── ...
│   └── tgt.csv         # CSV with image filenames and labels
└── val/
    ├── images/         # Directory containing validation images
    │   ├── image1.jpg
    │   └── ...
    └── tgt.csv         # CSV with image filenames and labels
```

The `tgt.csv` should contain image names and their corresponding text labels.

## Data Details

### Character Normalization

To create a uniform character set for the OCR model, the following text normalizations are applied to the labels:

| Original Character(s) | Description        | Normalized Character |
| :-------------------- | :----------------- | :------------------- |
| `“`, `”`             | Smart Quotes       | `"`                  |
| `’`                   | Typographical Apostrophe | `'`                  |
| `–`, `—`, `−`         | Various Dashes     | `-`                  |
| `…`                   | Ellipsis           | `...`                |
| `Ð`                   | Icelandic Eth (Uppercase) | `Đ`                  |
| `ð`                   | Icelandic Eth (Lowercase) | `đ`                  |
| `Ö`, `Ō`             | O with accents     | `O`                  |
| `Ü`, `Ū`             | U with accents     | `U`                  |
| `Ā`                   | A with macron      | `A`                  |
| `ö`, `ō`             | o with accents     | `o`                  |
| `ü`, `ū`             | u with accents     | `u`                  |
| `ā`                   | a with macron      | `a`                  |

This normalization simplifies the vocabulary the model needs to learn.


### Datasets Used

The training leverages a combined dataset from the following sources:

| Dataset                                                                                  | Train Samples | Validation Samples | Notes                       |
| :--------------------------------------------------------------------------------------- | --------------: | -------------------: | :-------------------------- |
| [vietocr](https://github.com/pbcquoc/vietocr)                                           | 441,025        | 110,257             | Random word images removed |
| [Paper (Deep Text Rec. Benchmark)](https://github.com/clovaai/deep-text-recognition-benchmark) | 3,287,346      | 6,992               |                             |
| [Synth90k](https://www.robots.ox.ac.uk/~vgg/data/text/)                                  | 7,224,612      | 802,731             |                             |
| [Cinnamon AI (Handwritten)](https://www.kaggle.com/datasets/hariwh0/cinnamon-ai-handwritten-addresses) | 1,470          | 368                 |                             |
| **Combined Total**                                                                       | **~11.0 M**    | **~0.9 M**          |                             |

**Vietnamese Data:** Please note that Vietnamese samples constitute only **1.76%** (209,120 images) of this combined dataset, from **VietOCR** (207,282) and **Cinnamon AI** (1,838). This reflects the limited availability of public Vietnamese OCR data.

## Training

Use the `cli.py` script with the `fit` command to start training. Key parameters can be adjusted via command-line arguments.

**Example Training Command (modify paths and parameters as needed):**

```bash
# Navigate to the script directory or adjust paths accordingly
# Example using parameters from the original README
# Found in script/train.sh

python cli.py fit \
    --data.train_data_path "/path/to/your/data/directory/train" \
    --data.val_data_path "/path/to/your/data/directory/val" \
    --data.batch_size 64 \
    --data.num_workers 8 \
    --data.dali True \
    --data.frac 1.0 \
    --model.backbone_name "resnet18" \
    --model.seq_name "bilstm" \
    --model.pred_name "attn" \
    --model.learning_rate 1e-4 \
    --model.batch_max_length 200 \
    --model.save_dir "checkpoints/my_experiment" \
    --trainer.max_epochs 10 \
    --trainer.val_check_interval 0.5 \
    --trainer.logger WandbLogger \
    --trainer.logger.name "my-ocr-run" \
    --trainer.logger.project "OCR_Project" \
    --trainer.log_every_n_steps 50
```

**Key Training Parameters:**

*   `--data.train_data_path`, `--data.val_data_path`: Paths to your training and validation data.
*   `--data.batch_size`, `--data.num_workers`: Configure data loading.
*   `--data.dali`: Set to `True` to use NVIDIA DALI (requires compatible hardware).
*   `--data.frac`: Use a fraction of the data (e.g., `0.1` for 10%).
*   `--model.backbone_name`: CNN feature extractor (`resnet18`, `vgg`, etc.).
*   `--model.seq_name`: Sequence model (`bilstm`, `none`).
*   `--model.pred_name`: Prediction head (`ctc`, `attn`).
*   `--model.learning_rate`: Optimizer learning rate.
*   `--model.batch_max_length`: Max sequence length for padding/processing.
*   `--model.save_dir`: Where to save model checkpoints.
*   `--trainer.*`: PyTorch Lightning trainer configurations (epochs, validation frequency, logger, etc.).

Refer to `python cli.py fit --help` for all available options.

## Inference

Use the `inference.py` script to run predictions on new images using a trained checkpoint.

**Example Inference Command:**

```bash
# Found in script/inference.sh
python inference.py \
    --image_path "/path/to/your/images_or_single_image.jpg" \
    --checkpoint "/path/to/your/checkpoints/my_experiment/model.ckpt"
```

**Inference Parameters:**

*   `--image_path`: Path to a single image file or a directory containing images.
*   `--checkpoint`: Path to the trained model checkpoint (`.ckpt`) file.

## Evaluation

Use the `test.py` script to evaluate a trained model's performance on standard OCR benchmark datasets.

**Example Evaluation Command:**

```bash
# Found in script/test.sh
python test.py \
    --checkpoint "/path/to/your/checkpoints/my_experiment/model.ckpt" \
    --data_root "/path/to/evaluation/datasets/root" \
    --output_dir "evaluation_results" \
    --batch_size 32
```

**Evaluation Parameters:**

*   `--checkpoint`: Path to the trained model checkpoint (`.ckpt`) file.
*   `--data_root`: The root directory containing benchmark datasets (e.g., IIIT5k, SVT, IC13, etc.). The script expects specific subdirectories for each benchmark.
*   `--output_dir`: Directory to save evaluation results.
*   `--batch_size`: Batch size for evaluation.

The script will automatically detect and run evaluation on supported datasets found within the `--data_root`, such as:
CUTE80, IC03_860, IC03_867, IC13_1015, IC13_857, IC15_1811, IC15_2077, IIIT5k_3000, SVT, SVTP.