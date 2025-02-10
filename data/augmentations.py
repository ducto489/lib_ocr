import torchvision.transforms as transforms


class Scaling:
    def __call__(self, image):
        w, h = image.size
        W, H = (420, 100)
        scale_ratio = min(W/w, H/h)
        return transforms.functional.resize(image,(int(h*scale_ratio), int(w*scale_ratio)))

data_transforms = {
    "train": transforms.Compose(
        [
            # transforms.Resize((100, 420)),
            Scaling(),
            # If the size is smaller than (100, 420), it will padding. Else it will crop. But above I have assertion if any image have size bigger than (100, 420)
            transforms.CenterCrop((100, 420)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "val": transforms.Compose(
        [
            Scaling(),
            transforms.CenterCrop((100, 420)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}