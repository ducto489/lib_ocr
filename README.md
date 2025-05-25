# OCR - Optical Character Recognition Engine

This repository provides a flexible and high-performance Optical Character Recognition (OCR) system built with PyTorch Lightning and accelerated using NVIDIA DALI for data loading.

Check out my detailed documentation to learn about data handling, training approaches, and the performance benefits of NVIDIA DALI: **[Accelerating OCR Training with NVIDIA DALI: A Practical Guide and Case Study](https://ducto489.github.io/projects/ocr-dali/)**


![clickbait image](/image/Pytorch-Dataloader.png)

_**Left (PyTorch DataLoader):** The GPU frequently idles or is underutilized, indicating data bottlenecks. **Right (NVIDIA DALI):** The GPU maintains consistently high utilization. DALI keeps the L4 GPU working hard, reducing wasted cycles and speeding up training._

## Overview

This project implements an end-to-end OCR pipeline, featuring:

*   **Multiple Architectures:** Easily swap CNN backbones (ResNet, VGG) and sequence modeling layers (BiLSTM).
*   **Prediction Methods:** Supports both CTC (Connectionist Temporal Classification) and Attention-based decoding.
*   **High-Performance Data Loading:** Integrates NVIDIA DALI for significantly faster data preprocessing and loading, especially beneficial on NVIDIA GPUs.
*   **Structured Training:** Leverages PyTorch Lightning for organized, reproducible training workflows, including logging (e.g., Wandb) and checkpointing.

## Table of Contents

- [Acknowledge](#acknowledgements)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

## Acknowledgements

Special thanks to **Trong Tuan** ([@santapo](https://github.com/santapo)) and **Phuong** ([@mp1704](https://github.com/mp1704)) for their significant help to this project.

## Setup

Follow these steps to set up your environment and download the necessary data.

### Get the Code

```bash
git clone https://github.com/AnyGlow/lib_ocr.git
cd lib_ocr
```

### Install Dependencies

```bash
conda create -n ocr python=3.11
conda activate ocr

pip install -r requirements.txt
```

### Download Data

The primary dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/ducto489/ocr_datasets):

```bash
pip install huggingface_hub

huggingface-cli download ducto489/ocr_datasets ocr_dataset.zip --repo-type dataset --local-dir .
unzip ocr_dataset.zip -d /path/to/your/data/directory
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

## Training

Use the `cli.py` script with the `fit` command to start training. Key parameters can be adjusted via command-line arguments.

**Example Training Command (modify paths and parameters as needed):**

```bash
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
```

**Key Training Parameters:**

*   `--data.train_data_path`, `--data.val_data_path`: Paths to your training and validation data.
*   `--data.batch_size`, `--data.num_workers`: Configure data loading.
*   `--data.dali`: Set to `True` to use NVIDIA DALI.
*   `--data.frac`: Use a fraction of the data (e.g., `0.1` for 10%).
*   `--model.backbone_name`: CNN feature extractor (`resnet18`, `vgg`).
*   `--model.seq_name`: Sequence model (`bilstm`, `none`).
*   `--model.pred_name`: Prediction head (`ctc`, `attn`).
*   `--model.learning_rate`: Optimizer learning rate.
*   `--model.batch_max_length`: Max sequence length for padding/processing.
*   `--model.save_dir`: Where to save model checkpoints.
*   `--trainer.*`: PyTorch Lightning trainer configurations (epochs, validation frequency, logger, etc.).

Refer to `python cli.py fit --help` for all available options.

## Inference

Checkout the Jupyter notebook `inference/inference.ipynb` to run predictions on new images using a trained checkpoint from [Hugging Face](https://huggingface.co/ducto489/ocr_model).

| Backbone                                | Time          |
|-----------------------------------------|---------------|
| Our model: Resnet - Bilstm - Attention  | 73ms @ A6000  |
| VietOCR: VGG19-bn - Transformer         | 565ms @ A6000 |
| VietOCR: VGG19-bn - Seq2Seq             |  30ms @ A6000 |

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
