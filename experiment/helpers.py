import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    # Transpose the list of lists to stack by row instead of by column
    # This effectively swaps rows and columns
    imgs_transposed = []
    for col_idx in range(len(imgs[0])):
        new_row = []
        for row_idx in range(len(imgs)):
            if col_idx < len(imgs[row_idx]):  # Check if this column exists in this row
                new_row.append(imgs[row_idx][col_idx])
        if new_row:  # Only add non-empty rows
            imgs_transposed.append(new_row)

    # If we have no valid rows after transposition, return
    if not imgs_transposed:
        return

    num_rows = len(imgs_transposed)
    num_cols = len(imgs_transposed[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx, row in enumerate(imgs_transposed):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None and len(row_title) == num_rows:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()