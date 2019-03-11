from torchvision.datasets import ImageFolder


class Dataset(ImageFolder):
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        path, _ = self.samples[idx]
        return image, target, path
