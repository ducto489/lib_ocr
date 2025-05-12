#!/usr/bin/env python
# OCR Inference Script

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from main import OCRModel
from dataloader import DALI_OCRDataModule

eval_data_path = [
    "CUTE80_png",
    "IC03_860_png",
    "IC03_867_png",
    "IC13_1015_png",
    "IC13_857_png",
    "IC15_1811_png",
    "IC15_2077_png",
    "IIIT5k_3000_png",
    "SVT_png",
    "SVTP_png"
]

def parse_args():
    parser = argparse.ArgumentParser(description='OCR Inference Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing evaluation datasets')
    parser.add_argument('--batch_max_length', type=int, default=25, help='Maximum sequence length for text')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    return parser.parse_args()

def evaluate_dataset(model, dataset_path, args, device):
    # Initialize DALI_OCRDataModule for this dataset
    data_module = DALI_OCRDataModule(
        batch_max_length=args.batch_max_length,
        frac=1.0,
        dali=True,
        train_data_path=dataset_path,
        val_data_path=dataset_path,
        batch_size=args.batch_size,
        num_workers=4,
        pred_name=model.pred_name
    )
    
    # Get validation dataloader for inference
    val_dataloader = data_module.val_dataloader()
    
    # Process images and gather results
    results = []
    n_correct = 0
    total_samples = 0
    confidence_score_list = []
    
    for batch in val_dataloader:
        images = batch["data"]
        labels = batch["label"]
        
        # Move to device
        images = images.to(device)
        
        with torch.no_grad():
            # Forward pass based on prediction type
            if model.pred_name == "ctc":
                logits = model(images, text=labels)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                preds = log_probs.argmax(2).permute(1, 0).detach().cpu()
                text = data_module.converter.decode(preds, None)
            else:  # Attention model
                preds = model(images, text=labels[:, :-1]).to(device)
                pred_size = torch.LongTensor([preds.size(1)] * preds.size(0))
                _, pred_index = preds.max(2)
                text = data_module.converter.decode(pred_index, pred_size)
            
            # Calculate confidence scores
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            # Get ground truth
            ground_truth = data_module.converter.decode(labels, None)
            
            # Process predictions
            for i, (pred, gt, pred_max_prob) in enumerate(zip(text, ground_truth, preds_max_prob)):
                total_samples += 1
                
                if model.pred_name == "attn":
                    # Handle end of sentence token for attention model
                    eos_idx = '[EOS]'
                    if eos_idx in gt:
                        gt = gt[:gt.find(eos_idx)]
                    if eos_idx in pred:
                        pred_EOS = pred.find(eos_idx)
                        pred = pred[:pred_EOS]
                        pred_max_prob = pred_max_prob[:pred_EOS]
                
                if pred == gt:
                    n_correct += 1
                
                # Calculate confidence score
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case
                
                confidence_score_list.append(confidence_score)
                results.append(f"{gt}\t{pred}\t{confidence_score.item():.4f}")
    
    # Calculate overall accuracy
    accuracy = n_correct / float(total_samples) * 100 if total_samples > 0 else 0
    avg_confidence = sum(confidence_score_list)/len(confidence_score_list) if confidence_score_list else 0
    
    return {
        "results": results,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "total_samples": total_samples,
        "correct_samples": n_correct
    }

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    model = OCRModel.load_from_checkpoint(
        args.checkpoint,
        strict=False,
        batch_max_length=args.batch_max_length,
        dali=True,
        map_location=device,
        pred_name="attn"
    )
    model.eval()
    model.to(device)
    
    # Prepare to collect overall statistics
    all_results = {}
    total_correct = 0
    total_samples = 0
    
    # Evaluate each dataset
    for dataset in eval_data_path:
        dataset_path = os.path.join(args.data_root, dataset)
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path {dataset_path} does not exist. Skipping.")
            continue
            
        print(f"\nEvaluating dataset: {dataset}")
        
        # Run evaluation
        eval_results = evaluate_dataset(model, dataset_path, args, device)
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{dataset}_results.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('Ground Truth\tPrediction\tConfidence\n')
            f.write('\n'.join(eval_results["results"]))
        
        # Print statistics
        print(f"Dataset: {dataset}")
        print(f"  Accuracy: {eval_results['accuracy']:.2f}%")
        print(f"  Average confidence: {eval_results['avg_confidence']:.4f}")
        print(f"  Samples: {eval_results['total_samples']}")
        print(f"  Results saved to {output_file}")
        
        # Store results for summary
        all_results[dataset] = eval_results
        total_correct += eval_results["correct_samples"]
        total_samples += eval_results["total_samples"]
    
    # Calculate and print overall statistics
    if total_samples > 0:
        overall_accuracy = (total_correct / total_samples) * 100
        print("\n==== Overall Results ====")
        print(f"Total accuracy: {overall_accuracy:.2f}%")
        print(f"Total samples: {total_samples}")
        
        # Save summary to file
        summary_file = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Dataset\tAccuracy\tSamples\n")
            for dataset, results in all_results.items():
                f.write(f"{dataset}\t{results['accuracy']:.2f}%\t{results['total_samples']}\n")
            f.write(f"\nOverall\t{overall_accuracy:.2f}%\t{total_samples}\n")
        
        print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
