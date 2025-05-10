#!/usr/bin/env python3
"""
Image Dataset Cleaner

This script scans through all images in a dataset directory, attempts to load each one,
and deletes any corrupted images that cause errors. It also updates the tgt.csv file
to remove entries for deleted images.

Usage:
    python clean_image_dataset.py --data_path /path/to/dataset

The script expects a directory structure with:
- A tgt.csv file containing 'image_name' and 'label' columns
- Image files either directly in the data_path or in an 'images' subdirectory
"""

import os
import argparse
import pandas as pd
from PIL import Image
import io
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
logger.add("image_cleaner.log", rotation="10 MB")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean image dataset by removing corrupted images")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    return parser.parse_args()


def load_tgt_file(data_path):
    """Load the tgt.csv file with support for different encodings"""
    tgt_path = os.path.join(data_path, "tgt.csv")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Label file not found at {tgt_path}")

    # Try different encodings for text support
    encodings = ["utf-8", "utf-8-sig", "utf-16"]
    for encoding in encodings:
        try:
            df = pd.read_csv(tgt_path, encoding=encoding)
            logger.info(f"Successfully loaded tgt.csv with {encoding} encoding")
            return df, encoding
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                raise UnicodeDecodeError(f"Failed to read CSV with encodings: {encodings}")
            continue


def check_image(image_path):
    """Try to open and load an image, return True if successful, False otherwise"""
    try:
        with open(image_path, "rb") as f:
            file_bytes = f.read()

        bytes_io = io.BytesIO(file_bytes)
        with Image.open(bytes_io) as img:
            img.load()  # This will verify the image is valid
        return True
    except (OSError, IOError) as e:
        return False


def main():
    args = parse_args()
    data_path = args.data_path

    logger.info(f"Starting image dataset cleaning for: {data_path}")

    # Load the tgt.csv file
    try:
        df, encoding = load_tgt_file(data_path)
    except Exception as e:
        logger.error(f"Error loading tgt.csv: {str(e)}")
        return

    # Check required columns
    required_cols = ["image_name", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in CSV: {missing_cols}")
        return

    # Convert values to strings
    df["image_name"] = df["image_name"].astype(str)
    df["label"] = df["label"].astype(str)

    # Check if images are in the main directory or in an 'images' subdirectory
    sample_image = df["image_name"].iloc[0]
    images_dir = data_path
    if os.path.exists(os.path.join(data_path, "images", sample_image)):
        images_dir = os.path.join(data_path, "images")
        logger.info(f"Images found in subdirectory: {images_dir}")

    # Process each image
    total_images = len(df)
    corrupted_images = []

    logger.info(f"Checking {total_images} images...")

    for idx, row in df.iterrows():
        image_name = row["image_name"]
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            corrupted_images.append(image_name)
            continue

        if not check_image(image_path):
            logger.warning(f"Corrupted image found: {image_path}")
            corrupted_images.append(image_name)
            try:
                os.remove(image_path)
                logger.info(f"Deleted corrupted image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to delete {image_path}: {str(e)}")

    # Update the tgt.csv file
    if corrupted_images:
        logger.info(f"Found {len(corrupted_images)} corrupted images out of {total_images}")

        # Create backup of original file
        backup_path = os.path.join(data_path, "tgt.csv.bak")
        df.to_csv(backup_path, index=False, encoding=encoding)
        logger.info(f"Created backup of original tgt.csv at {backup_path}")

        # Remove corrupted images from dataframe
        df_clean = df[~df["image_name"].isin(corrupted_images)]

        # Save updated dataframe
        df_clean.to_csv(os.path.join(data_path, "tgt.csv"), index=False, encoding=encoding)
        logger.info(f"Updated tgt.csv, removed {len(corrupted_images)} entries")

        # Save list of corrupted images for reference
        with open(os.path.join(data_path, "corrupted_images.txt"), "w") as f:
            for img in corrupted_images:
                f.write(f"{img}\n")
        logger.info(f"Saved list of corrupted images to {os.path.join(data_path, 'corrupted_images.txt')}")
    else:
        logger.info("No corrupted images found. Dataset is clean!")


if __name__ == "__main__":
    main()
