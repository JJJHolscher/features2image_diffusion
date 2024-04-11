#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm
from jo3mnist.vis import to_img
from jo3mnist.tiny_imagenet import TinyImageTrainNet, TinyImageTestNet
from jo3mnist.imagenet import ImageNet
# from jo3util.warning import todo
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Resize


def load_imagenet_with_features(
    features_path,
    images_path,
    image_size=224,
    **kwargs
):
    train_loader = DataLoader(
        ImageNetSet(
            Path(features_path) / "train",
            Path(images_path) / "train",
            image_size=image_size
        ),
        **kwargs
    )
    test_loader = DataLoader(
        ImageNetSet(
            Path(features_path) / "validation",
            Path(images_path) / "validation",
            image_size=image_size
        ),
        **kwargs
    )
    return train_loader, test_loader


class ImageNetSet(Dataset):
    def __init__(self, features_path, imagenet_path, image_size=224):
        self.imagenet = ImageNet(imagenet_path, image_size=image_size)
        self.paths = [p for p in Path(features_path).glob("[0-9]*.npy")]

    def __getitem__(self, i):
        features = np.load(self.paths[i], allow_pickle=False).flatten()
        image, label = self.imagenet[i]
        return torch.Tensor(features), image, label

    def __len__(self):
        return len(self.paths)


class TinyImageNetSet(Dataset):
    resize = Resize([64, 64])

    def __init__(self, features_path, images_path, train: bool):
        if train:
            self.set = TinyImageTrainNet(Path(images_path))
            paths = Path(features_path).glob("train/[0-9]*.npy")
        else:
            self.set = TinyImageTestNet(Path(images_path))
            paths = Path(features_path).glob("test/[0-9]*.npy")
        self.paths = [p for p in paths]

        assert len(self.paths) == len(self.set)

    def __getitem__(self, i):
        features = np.load(self.paths[i], allow_pickle=False).flatten()
        image, label = self.set[i]
        image = self.resize(torch.Tensor(image))
        return torch.Tensor(features), image, label

    def __len__(self):
        return len(self.paths)


def load_tiny_imagenet_with_features(features_path, images_path, **kwargs):
    trainset = TinyImageNetSet(
        features_path,
        Path(images_path) / "train",
        train=True
    )
    testset = TinyImageNetSet(
        features_path,
        Path(images_path) / "val",
        train=False
    )
    trainloader = DataLoader(trainset, **kwargs)
    testloader = DataLoader(testset, **kwargs)
    return trainloader, testloader


class MnistFeaturesSet(Dataset):
    """The MNIST set that also yields the features associated with the image.

    When getting an item from this, it returns `features, image, label`
    """

    def __init__(self, features_path, mnist_path, train=True, transform=None):
        self.mnist = MNIST(
            mnist_path, train=train, download=True, transform=transform
        )

        train = "train" if train else "test"
        paths = Path(features_path).glob(train + "/[0-9]*.npy")
        self.paths = [p for p in paths]

        assert len(self.paths) == len(self.mnist)

    def __getitem__(self, i):
        features = np.load(self.paths[i], allow_pickle=False)
        image, label = self.mnist[i]
        return features, image, label

    def __len__(self):
        return len(self.paths)


def load_mnist_with_features(
    features_path: Union[Path, str],
    mnist_path: Union[Path, str],
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """ Load train and test dataloaders that yield features, image and label.
    """
    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = MnistFeaturesSet(
        features_path, mnist_path, train=True, transform=normalise_data
    )
    test_set = MnistFeaturesSet(
        features_path, mnist_path, train=False, transform=normalise_data
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


def loader_to_dataframe(data_loader, img_dir="./res/mnist/img"):
    data_len = len(data_loader)
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print("[0/1] storing mnist images")
        img_dir.mkdir(parents=True, exist_ok=True)
        iterator = tqdm(enumerate(data_loader), total=data_len)
        for i, (_, images, _) in iterator:
            for j, img in enumerate(images):
                index = i * len(images) + j
                to_img(img).save(img_dir / f"{index}.png")

    # Empty the data_loader iterator into tensors.
    print("[1/1] loading features and labels")
    all_features = []
    all_labels = []
    iterator = tqdm(enumerate(data_loader), total=data_len)
    for i, (features, _, labels) in iterator:
        all_features.append(torch.Tensor(features))
        all_labels.append(torch.Tensor(labels))

    dataframe = {"label": torch.concatenate(all_labels)}

    # Create a column per feature.
    all_features = torch.concatenate(all_features)
    feature_len = all_features.shape[-1]
    for f in range(feature_len):
        dataframe[f"f{f}"] = all_features[:, f]

    return pd.DataFrame(dataframe)
