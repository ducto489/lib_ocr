#!/usr/bin/env python
# OCR Inference Script

import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

from main import OCRModel
from dataloader import DALI_OCRDataModule
from utils import CTCLabelConverter_clovaai, AttnLabelConverter
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='OCR Inference Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to the image or directory of images')
    parser.add_argument('--batch_max_length', type=int, default=25, help='Maximum sequence length for text')
    parser.add_argument('--output', type=str, default='results.txt', help='Output file to save results')
    return parser.parse_args()

def load_model(checkpoint_path, batch_max_length):
    """Load the OCR model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate the model
    model = OCRModel.load_from_checkpoint(checkpoint_path, strict=False, batch_max_length=batch_max_length, 
                                          dali=False, map_location=device, pred_name="attn")
    model.eval()
    model.to(device)
    
    return model

def main():
    args = parse_args()
    
    # Load model
    model = load_model(args.checkpoint, args.batch_max_length)
    
    # Initialize DALI_OCRDataModule
    data_module = DALI_OCRDataModule(
        batch_max_length=args.batch_max_length,
        frac=1.0,  # Use all data
        dali=True,
        train_data_path=args.image,  # Use the input image path as data path
        val_data_path=args.image,  # Use the same path for validation
        batch_size=1,  # Process one image at a time
        num_workers=1,
        pred_name=model.pred_name
    )

    # Get the validation dataloader (we use this for inference)
    val_dataloader = data_module.val_dataloader()

    # Process each batch
    results = []
    n_correct = 0
    total_samples = 0
    confidence_score_list = []
    
    for batch in val_dataloader:
        images = batch["data"]
        labels = batch["label"]
        
        # Move to device
        images = images.to(model.device)
        
        with torch.no_grad():
            # Forward pass
            if model.pred_name == "ctc":
                logits = model(images, text=labels)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                preds = log_probs.argmax(2).permute(1, 0).detach().cpu()
                text = data_module.converter.decode(preds, None)
            else:  # Attention model
                preds = model(images, text=labels[:, :-1]).to(model.device)
                pred_size = torch.LongTensor([preds.size(1)] * preds.size(0))
                _, pred_index = preds.max(2)
                text = data_module.converter.decode(pred_index, pred_size)
        
        # Get ground truth
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        ground_truth = data_module.converter.decode(labels, None)
        
        # Calculate confidence scores and accuracy
        for i, (pred, gt, pred_max_prob) in enumerate(zip(text, ground_truth, preds_max_prob)):
            total_samples += 1
            
            if model.pred_name == "attn":
                # For attention model, prune after end of sentence token
                eos_idx = '[EOS]'
                if eos_idx in gt:
                    gt = gt[:gt.find(eos_idx)]
                if eos_idx in pred:
                    pred_EOS = pred.find(eos_idx)
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            # Calculate confidence score (multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case
            confidence_score_list.append(confidence_score)
            
            # Store result with confidence score
            results.append(f"{gt}\t{pred}\t{confidence_score.item():.4f}")
    
    # Calculate overall accuracy
    accuracy = n_correct / float(total_samples) * 100 if total_samples > 0 else 0
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average confidence score: {sum(confidence_score_list)/len(confidence_score_list):.4f}" if confidence_score_list else "No confidence scores calculated")

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('Ground Truth\tPrediction\tConfidence\n')
        f.write('\n'.join(results))
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
