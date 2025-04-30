import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for seaborn
sns.set(style="whitegrid")

# Read the CSV files
csv_file1 = 'no-dali.csv'
csv_file2 = 'dali-all-CPU-processing.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

file1 = csv_file1[:-4]
file2 = csv_file2[:-4]

# Extract the GPU utilization columns
gpu_utilization1 = df1['train50-' + file1 + ' - system/gpu.0.gpu']
gpu_utilization2 = df2['train50-' + file2 + ' - system/gpu.0.gpu']

# Print basic statistics for both datasets
print("Basic statistics for GPU utilization (with DALI " + file1 + "):")
print(f"Count: {gpu_utilization1.count()}")
print(f"Mean: {gpu_utilization1.mean():.2f}%")
print(f"Median: {gpu_utilization1.median():.2f}%")
print(f"Min: {gpu_utilization1.min():.2f}%")
print(f"Max: {gpu_utilization1.max():.2f}%")
print(f"Standard Deviation: {gpu_utilization1.std():.2f}%")

print("\nBasic statistics for GPU utilization (without DALI " + file2 + "):")
print(f"Count: {gpu_utilization2.count()}")
print(f"Mean: {gpu_utilization2.mean():.2f}%")
print(f"Median: {gpu_utilization2.median():.2f}%")
print(f"Min: {gpu_utilization2.min():.2f}%")
print(f"Max: {gpu_utilization2.max():.2f}%")
print(f"Standard Deviation: {gpu_utilization2.std():.2f}%")

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot histogram for the first dataset (with DALI)
sns.histplot(gpu_utilization1, bins=30, kde=True, color='blue', ax=ax1)
ax1.set_title('GPU Utilization Distribution ' + file1, fontsize=14)
ax1.set_xlabel('GPU Utilization (%)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.axvline(gpu_utilization1.mean(), color='red', linestyle='dashed', linewidth=2, 
           label=f'Mean: {gpu_utilization1.mean():.2f}%')
ax1.axvline(gpu_utilization1.median(), color='green', linestyle='dashed', linewidth=2, 
           label=f'Median: {gpu_utilization1.median():.2f}%')
ax1.legend()

# Plot histogram for the second dataset (without DALI)
sns.histplot(gpu_utilization2, bins=30, kde=True, color='orange', ax=ax2)
ax2.set_title('GPU Utilization Distribution ' + file2, fontsize=14)
ax2.set_xlabel('GPU Utilization (%)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.axvline(gpu_utilization2.mean(), color='red', linestyle='dashed', linewidth=2, 
           label=f'Mean: {gpu_utilization2.mean():.2f}%')
ax2.axvline(gpu_utilization2.median(), color='green', linestyle='dashed', linewidth=2, 
           label=f'Median: {gpu_utilization2.median():.2f}%')
ax2.legend()

plt.tight_layout()
plt.savefig('gpu_utilization_comparison_' + file1 + '_' + file2 + '.png', dpi=300)
print("Comparison histogram saved as 'gpu_utilization_comparison.png'")

# Create a single plot with both distributions for direct comparison
plt.figure(figsize=(12, 6))
sns.histplot(gpu_utilization1, bins=30, kde=True, color='blue', alpha=0.6, label=file1)
sns.histplot(gpu_utilization2, bins=30, kde=True, color='orange', alpha=0.6, label=file2)
plt.title('GPU Utilization Comparison ' + file1 + ' ' + file2, fontsize=16)
plt.xlabel('GPU Utilization (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gpu_utilization_overlay_' + file1 + '_' + file2 + '.png', dpi=300)
print("Overlay histogram saved as 'gpu_utilization_overlay.png'")

# Show the plots
plt.show()
