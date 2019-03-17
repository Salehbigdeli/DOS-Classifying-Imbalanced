from PIL import Image
from torchvision.datasets import ImageFolder
import torch.utils.data as data


class Dataset(ImageFolder):
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        path, _ = self.samples[idx]
        return image, target, path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImbalancedDataset(data.Dataset):

    def __init__(self, fnames, transform, target_transform, loader, distances_matrix,
                 class_wise_overloading, class_wise_oversampling, class_neighbors):
        if loader is None:
            self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.fnames = []
        self.classes = []
        for cls, names in fnames.items():
            self.fnames.extend(names)
            self.classes.extend([cls]*len(names))
        self.distances_matrix = distances_matrix
        self.class_wise_overloading = class_wise_overloading
        self.class_wise_oversampling = class_wise_oversampling
        self.class_neighbors = class_neighbors

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, item):
        path, target = self.fnames[item], self.classes[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
