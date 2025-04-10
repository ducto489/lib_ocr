import cv2
import json
import os
from PIL import Image
import math
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd


class OCRDataset(Dataset):
    def __init__(self, data_path, transform=None, batch_max_length=50):
        self.data_path = data_path
        self.transform = transform
        self.batch_max_length = batch_max_length
        
        images, labels = process_tgt(data_path, batch_max_length)
        self.data = list(zip(images, labels)) #list(zip(df['image_name'], df['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Try to get a valid image, if current idx fails, try the next one
        original_idx = idx
        max_retries = 10  # Limit retries to avoid infinite loops
        retries = 0
        
        while retries < max_retries:
            try:
                image_name, label = self.data[idx]
                image_path = os.path.join(self.data_path, 'images', image_name)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                    
                # Use PIL's Image.open with error handling for truncated images
                try:
                    image = Image.open(image_path).convert('RGB')
                except (OSError, IOError) as e:
                    print(f"Warning: Skipping corrupted image {image_path}: {str(e)}")
                    idx = (idx + 1) % len(self.data)  # Try next image
                    retries += 1
                    continue
                
                if self.transform:
                    try:
                        image = self.transform(image)
                    except Exception as e:
                        print(f"Transform failed for {image_path}: {str(e)}")
                        idx = (idx + 1) % len(self.data)  # Try next image
                        retries += 1
                        continue
                        
                return image, label
                
            except (IOError, FileNotFoundError) as e:
                print(f"Error loading image at index {idx}: {str(e)}")
                idx = (idx + 1) % len(self.data)  # Try next image
                retries += 1
        
        # If we've tried max_retries times and still failed, raise an exception
        raise RuntimeError(f"Failed to load a valid image after {max_retries} attempts starting from index {original_idx}")


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0) 
        Pad_img[:, :, :w] = img  # right pad
        
        if self.max_size[2] != w:  # add the same background for new padding
            temp = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            temp_1 = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            print(temp.shape)
            print(temp_1.shape)
        #     Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2)
            

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        # import pdb; pdb.set_trace()
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images] 
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels
    
def process_tgt(data_path, batch_max_length):
        # Check for tgt.csv file
    tgt_path = os.path.join(data_path, 'tgt.csv')
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Label file not found at {tgt_path}")
    
    # Try different encodings for Vietnamese text support
    encodings = ['utf-8', 'utf-8-sig', 'utf-16']
    for encoding in encodings:
        try:
            df = pd.read_csv(tgt_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                raise UnicodeDecodeError(f"Failed to read CSV with encodings: {encodings}")
            continue
    
    # Validate required columns
    required_cols = ['image_name', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    # Convert values to strings and filter by length
    df['image_name'] = df['image_name'].astype(str)
    df['label'] = df['label'].astype(str)
    
    # Filter out samples exceeding batch_max_length
    total_samples = len(df)
    df = df[df['label'].str.len() <= batch_max_length]#.sample(frac=0.01, random_state=42)
    filtered_samples = total_samples - len(df)
    if filtered_samples > 0:
        print(f"Filtered out {filtered_samples} samples ({filtered_samples/total_samples*100:.2f}%) exceeding max length {batch_max_length}")
        
    return list(df['image_name']), list(df['label'])