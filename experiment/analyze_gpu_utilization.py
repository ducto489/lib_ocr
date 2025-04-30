import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
csv_file = '/hdd1t/mduc/ocr/lib_ocr/experiment/wandb_export_2025-04-30T17_37_05.262+07_00.csv'
df = pd.read_csv(csv_file)

# Extract the GPU utilization column
gpu_utilization = df['train50-dali - system/gpu.0.gpu']

# Print basic statistics
print("Basic statistics for GPU utilization:")
print(f"Count: {gpu_utilization.count()}")
print(f"Mean: {gpu_utilization.mean():.2f}%")
print(f"Median: {gpu_utilization.median():.2f}%")
print(f"Min: {gpu_utilization.min():.2f}%")
print(f"Max: {gpu_utilization.max():.2f}%")
print(f"Standard Deviation: {gpu_utilization.std():.2f}%")

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(gpu_utilization, bins=100, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of GPU Utilization (train50-dali)', fontsize=16)
plt.xlabel('GPU Utilization (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', alpha=0.75)

# Add a vertical line for the mean
plt.axvline(gpu_utilization.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {gpu_utilization.mean():.2f}%')

# Add a vertical line for the median
plt.axvline(gpu_utilization.median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {gpu_utilization.median():.2f}%')

plt.legend()
# plt.tight_layout()

# Save the histogram
plt.savefig('gpu_utilization_dali_histogram.png', dpi=300)
print("Histogram saved as 'gpu_utilization_histogram.png'")

# Show the histogram
plt.show()
