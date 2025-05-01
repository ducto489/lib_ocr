import csv
import os
import webdataset as wds
import io # For handling potential encoding issues

# --- Configuration ---
csv_file_path = '/home/qsvm/dataset/val/tgt.csv' # Your input CSV file
output_dir = '/home/qsvm/webdataset_2/val' # Where to save the .tar files
shard_pattern = os.path.join(output_dir, 'ocr-shard-%06d.tar') # Naming pattern for shards
max_samples_per_shard = 30000 # How many samples (image+label pairs) per shard? Adjust as needed.
# --- End Configuration ---

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Use ShardWriter to automatically handle splitting into multiple TAR files
# Set verbose=0 to make the shard writing process silent
with wds.ShardWriter(shard_pattern, maxcount=max_samples_per_shard, verbose=0) as shard_writer:
    # Open and read the CSV file
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile) # Assumes header row: image_path,label

        for i, row in enumerate(reader):
            image_path = '/home/qsvm/dataset/val/images/' + row['image_name']
            label_text = row['label']

            # Create a unique base key for this sample
            # Example: If image_path is path/to/images/img_0001.png, key might be img_0001
            base_key = os.path.splitext(os.path.basename(image_path))[0]

            # Add suffix to make it unique if base names clash (optional but safer)
            sample_key = f"{base_key}_{i:08d}"

            try:
                # Read image data as bytes
                with open(image_path, "rb") as stream:
                    image_data = stream.read()

                # Encode label text as bytes (UTF-8 is common)
                label_data = label_text.encode('utf-8')

                # Determine the image extension (important for DALI decoder)
                _, image_extension = os.path.splitext(image_path)
                image_extension = image_extension.lower().lstrip('.') # e.g., 'jpg', 'png'

                # Create the sample dictionary
                # Keys: '__key__', plus one key per data component (use file extensions)
                sample = {
                    "__key__": sample_key,
                    image_extension: image_data, # e.g., "jpg": image_bytes
                    "txt": label_data          # Store label as ".txt"
                }

                # Write the sample to the current shard
                shard_writer.write(sample)

                # if (i + 1) % 1000 == 0:
                #     print(f"Processed {i+1} samples...")

            except FileNotFoundError:
                print(f"Warning: Image file not found, skipping: {image_path}")
            except Exception as e:
                print(f"Error processing row {i+1} (key: {sample_key}): {e}")

print("Finished creating WebDataset shards.")