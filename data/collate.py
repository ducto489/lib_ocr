import torch
from torchvision import transforms


class OCRCollator:

    def __init__(self):
        self.to_tensor = transforms.ToTensor()  # Convert PIL image to tensor

    def __call__(self, batch):
        # Unpack batch into separate lists
        images, labels = zip(*batch)

        # Convert images to tensors and get their shapes
        images = [self.to_tensor(img) for img in images]
        channels, height = images[0].shape[0], images[0].shape[1]
        widths = [img.shape[2] for img in images]

        # Create a tensor to hold the padded images
        max_width = max(widths)
        padded_imgs = torch.zeros((len(images), channels, height, max_width),
                                  dtype=torch.float32)

        # Copy each image into the tensor with padding
        for i, img in enumerate(images):
            orig_width = widths[i]
            padded_imgs[i, :, :, :orig_width] = img
            # add the same background for new padding
            # if not, the added background will be black
            if orig_width < max_width:
                padded_imgs[i, :, :, orig_width:] = img[:, :, -1].unsqueeze(2)

        return {
            "images": padded_imgs,  # Batched image tensor
            "labels": labels,  # List of labels (strings)
        }
