from torchvision.transforms import v2


class Scaling:
    def __call__(self, image):
        w, h = image.size
        H = 100
        scale_ratio = H/h
        return v2.functional.resize(image,(100, int(w*scale_ratio)))

class AlbumentationsScaling:
    def __init__(self, height=100):
        self.height = height

    def __call__(self, image):
        h, w = image.shape[:2]  # Albumentations uses numpy arrays (HWC format)
        scale_ratio = self.height / h
        new_width = int(w * scale_ratio)
        return A.resize(image, height=self.height, width=new_width)

data_transforms = {
    "train": v2.Compose(
        [
            Scaling(),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ]
    ),
    "val": v2.Compose(
        [
            Scaling(),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,)),
        ]
    ),
}

# TODO: Is the scale factor correct? Do I need to make it smaller?
data_transforms_2 = {
    "train": v2.Compose(
        [
            Scaling(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomAffine(degrees=5, scale=(0.9, 1.1)),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]
    ),
    "val": v2.Compose(
        [
            Scaling(),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]
    ),
}