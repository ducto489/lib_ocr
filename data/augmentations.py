from torchvision.transforms import v2


class Scaling:
    def __call__(self, image):
        w, h = image.size
        H = 100
        scale_ratio = H / h
        return v2.functional.resize(image, (100, int(w * scale_ratio)))


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

data_transforms_2 = {
    "train": v2.Compose(
        [
            Scaling(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.3),
            v2.RandomAffine(degrees=2, scale=(0.9, 1)),
            v2.RandomPerspective(distortion_scale=0.1, p=0.5),
            v2.ToTensor(),
            v2.GaussianNoise(),
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
